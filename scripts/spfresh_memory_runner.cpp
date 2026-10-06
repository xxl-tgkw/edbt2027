#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/SPFresh/SPFresh.h"

// Build and update the official SPFresh code path in one process.  The
// SPFRESH_SPDK_USE_MEM_IMPL=1 backend is an in-memory SPDK-compatible
// fallback; it must not be interpreted as NVMe latency.
int main(int argc, char** argv) {
    if (argc != 2) {
        LOG(SPTAG::Helper::LogLevel::LL_Error,
            "spfresh_memory_runner config.ini\n");
        return 2;
    }
    SPTAG::Helper::IniReader ini;
    if (ini.LoadIniFile(argv[1]) != SPTAG::ErrorCode::Success) return 3;
    auto index = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::SPANN, SPTAG::VectorValueType::Float);
    const char* sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (const char* section : sections) {
        for (const auto& kv : ini.GetParameters(section)) {
            index->SetParameter(kv.first, kv.second, section);
        }
    }
    if (index->BuildIndex() != SPTAG::ErrorCode::Success) {
        LOG(SPTAG::Helper::LogLevel::LL_Error, "SPFresh build failed\n");
        return 4;
    }
    auto* spann = static_cast<SPTAG::SPANN::Index<float>*>(index.get());
    LOG(SPTAG::Helper::LogLevel::LL_Info,
        "SPFresh native memory build complete: vectors=%d\n",
        spann->GetNumSamples());
    SPTAG::SSDServing::SPFresh::UpdateSPFresh(spann);
    LOG(SPTAG::Helper::LogLevel::LL_Info,
        "SPFresh native memory update complete: vectors=%d\n",
        spann->GetNumSamples());
    return 0;
}
