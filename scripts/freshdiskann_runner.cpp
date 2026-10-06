// Native Fresh-DiskANN dynamic-memory runner for UVR-Bench.
//
// This intentionally uses the upstream DiskANN cpp_main API.  The runner keeps
// vectors and graph in memory, inserts one update round at a time, and emits a
// line-oriented log that can be audited without any post-hoc timing synthesis.

#include <index.h>
#include <index_factory.h>
#include <program_options_utils.hpp>
#include <timer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using diskann::IndexConfig;
using diskann::IndexConfigBuilder;
using diskann::IndexFactory;
using diskann::IndexSearchParams;
using diskann::IndexWriteParameters;

struct FBin {
    uint32_t n = 0, d = 0;
    std::vector<float> x;
};

static FBin read_fbin(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    FBin a;
    in.read(reinterpret_cast<char *>(&a.n), sizeof(a.n));
    in.read(reinterpret_cast<char *>(&a.d), sizeof(a.d));
    a.x.resize(static_cast<size_t>(a.n) * a.d);
    in.read(reinterpret_cast<char *>(a.x.data()), static_cast<std::streamsize>(a.x.size() * sizeof(float)));
    if (!in) throw std::runtime_error("short fbin " + path);
    return a;
}

static std::vector<uint32_t> read_truth(const std::string &path, uint32_t &rows, uint32_t &k) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    uint32_t n;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    in.read(reinterpret_cast<char *>(&k), sizeof(k));
    if (k == 0 || n % k != 0) throw std::runtime_error("bad truth shape");
    rows = n / k;
    std::vector<uint32_t> t(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(uint32_t)));
    if (!in) throw std::runtime_error("short truth");
    return t;
}

static double recall10(const std::vector<uint32_t> &got, const uint32_t *truth, uint32_t k) {
    uint32_t hit = 0;
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < got.size(); ++j)
            // The upstream dynamic API reserves tag 0; protocol ids are zero based.
            if (got[j] == truth[i] + 1u) { ++hit; break; }
    return static_cast<double>(hit) / static_cast<double>(k);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: freshdiskann_runner VECTORS QUERIES TRUTH OUT BASE BATCH\n";
        return 2;
    }
    const std::string vectors_path = argv[1], queries_path = argv[2], truth_path = argv[3], out_path = argv[4];
    const uint32_t base = static_cast<uint32_t>(std::stoul(argv[5]));
    const uint32_t batch = static_cast<uint32_t>(std::stoul(argv[6]));
    const uint32_t rounds = 5, nq = 128, k = 10, threads = 8, R = 64, Lbuild = 100;
    const uint32_t budgets[] = {64, 128, 256, 512};

    FBin vec = read_fbin(vectors_path), qry = read_fbin(queries_path);
    uint32_t truth_rows = 0, truth_k = 0;
    std::vector<uint32_t> truth = read_truth(truth_path, truth_rows, truth_k);
    if (vec.d != qry.d || vec.n < base + rounds * batch || qry.n != nq || truth_rows != 6 * nq || truth_k != k)
        throw std::runtime_error("input shape does not match protocol");

    // DiskANN pads dimensions to a multiple of eight in its aligned data store.
    const uint32_t ad = static_cast<uint32_t>((vec.d + 7u) & ~7u);
    std::vector<float> aligned(static_cast<size_t>(vec.n) * ad, 0.0f);
    for (uint32_t i = 0; i < vec.n; ++i)
        std::copy(vec.x.begin() + static_cast<size_t>(i) * vec.d,
                  vec.x.begin() + static_cast<size_t>(i + 1) * vec.d,
                  aligned.begin() + static_cast<size_t>(i) * ad);
    std::vector<float> aligned_q(static_cast<size_t>(qry.n) * ad, 0.0f);
    for (uint32_t i = 0; i < qry.n; ++i)
        std::copy(qry.x.begin() + static_cast<size_t>(i) * qry.d,
                  qry.x.begin() + static_cast<size_t>(i + 1) * qry.d,
                  aligned_q.begin() + static_cast<size_t>(i) * ad);

    IndexWriteParameters wp = diskann::IndexWriteParametersBuilder(Lbuild, R)
                                  .with_alpha(1.2f).with_saturate_graph(false)
                                  .with_num_threads(threads).build();
    IndexSearchParams sp(Lbuild, threads);
    IndexConfig cfg = IndexConfigBuilder()
                          .with_metric(diskann::L2).with_dimension(vec.d)
                          .with_max_points(vec.n).is_dynamic_index(true)
                          .with_index_write_params(wp).with_index_search_params(sp)
                          .with_data_type("float").with_tag_type("uint32")
                          .with_label_type("uint32").with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .is_enable_tags(true).is_filtered(false).with_num_frozen_pts(1)
                          .is_concurrent_consolidate(false).build();
    IndexFactory factory(cfg);
    std::unique_ptr<diskann::AbstractIndex> index = factory.create_instance();
    index->set_start_points_at_random(1.0f, 0);

    std::vector<uint32_t> tags(base);
    std::iota(tags.begin(), tags.end(), 1u);
    auto t0 = std::chrono::steady_clock::now();
    index->build(aligned.data(), base, tags);
    double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    out << "#protocol=freshdiskann-native-memory-v1\n#dimension=" << vec.d << "\n#base=" << base
        << "\n#batch=" << batch << "\n#rounds=" << rounds << "\n#build_ms=" << std::setprecision(12) << build_ms << "\n";
    for (uint32_t r = 1; r <= rounds; ++r) {
        const uint32_t lo = base + (r - 1) * batch, hi = lo + batch;
        t0 = std::chrono::steady_clock::now();
        for (uint32_t i = lo; i < hi; ++i) {
            int rc = index->insert_point(aligned.data() + static_cast<size_t>(i) * ad, i + 1u);
            if (rc != 0) throw std::runtime_error("insert_point failed at " + std::to_string(i));
        }
        const double insert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out << "insert," << r << "," << std::setprecision(12) << insert_ms << "\n";
        for (uint32_t b : budgets) {
            std::vector<double> rec(nq), ms(nq);
            for (uint32_t qi = 0; qi < nq; ++qi) {
                std::vector<uint32_t> ids(k);
                std::vector<float> dist(k);
                auto q0 = std::chrono::steady_clock::now();
                std::vector<float *> result_vectors;
                index->search_with_tags(aligned_q.data() + static_cast<size_t>(qi) * ad, k, b, ids.data(), dist.data(),
                                        result_vectors);
                ms[qi] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q0).count();
                rec[qi] = recall10(ids, truth.data() + (static_cast<size_t>(r) * nq + qi) * k, k);
                out << "query," << r << "," << b << "," << qi << "," << std::setprecision(12) << rec[qi]
                    << "," << ms[qi] << "\n";
            }
        }
        out.flush();
    }
    std::cout << "build_ms=" << build_ms << " log=" << out_path << "\n";
    return 0;
}
