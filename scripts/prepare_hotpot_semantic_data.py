#!/usr/bin/env python3
"""Freeze a compact, public HotpotQA document-retrieval vector workload.

The source is the cached Hugging Face HotpotQA distractor validation split.  We
encode context sentences and questions with a local, pinned
all-MiniLM-L6-v2 model.  The resulting pool is intentionally a benchmark
fixture, not an agent trajectory or a claim about production timestamps.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/semantic_hotpot"
SOURCE = Path("/home/ubuntu/.cache/huggingface/hub/datasets--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet")
MODEL = Path("/home/ubuntu/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
SEED = 20261004
N_QUERIES = 512
FINAL = 100_000
BASE = 80_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_sha(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Expected cached source: {SOURCE}")
    if not MODEL.exists():
        raise FileNotFoundError(f"Expected cached encoder: {MODEL}")
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "pool.npz"
    manifest_path = DATA / "manifest.json"
    if out.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha(out) != manifest["pool_sha256"]:
            raise ValueError("Existing Hotpot pool checksum mismatch")
        print(json.dumps(manifest, indent=2))
        return

    table = pq.read_table(SOURCE)
    rows = table.to_pylist()
    rng = np.random.default_rng(SEED)
    candidates = []
    candidate_keys = set()
    supports_by_q = []
    questions = []

    # Keep sentence boundaries and titles so supporting-fact annotations remain
    # auditable after vectorization.  Identical title/sentence pairs are deduped.
    for row in rows:
        context = row["context"]
        title_to_sent = {}
        for title, sentences in zip(context["title"], context["sentences"]):
            title_to_sent[title] = list(sentences)
            for sent_id, sentence in enumerate(sentences):
                text = f"{title}: {sentence}".strip()
                key = (title, int(sent_id), sentence)
                if key not in candidate_keys:
                    candidate_keys.add(key)
                    candidates.append({"title": title, "sent_id": int(sent_id), "text": text, "support": 0})
        supports = []
        facts = row["supporting_facts"]
        for title, sent_id in zip(facts["title"], facts["sent_id"]):
            supports.append((title, int(sent_id)))
        questions.append(row["question"])
        supports_by_q.append(supports)

    if len(questions) < N_QUERIES or len(candidates) < FINAL:
        raise ValueError(f"Insufficient source rows/candidates: {len(questions)}, {len(candidates)}")

    qids = np.sort(rng.choice(len(questions), size=N_QUERIES, replace=False))
    selected_questions = [questions[int(i)] for i in qids]
    support_keys = set()
    for qi in qids:
        support_keys.update(supports_by_q[int(qi)])
    support_indices = [i for i, item in enumerate(candidates) if (item["title"], item["sent_id"]) in support_keys]
    for i in support_indices:
        candidates[i]["support"] = 1
    if len(support_indices) > FINAL // 5:
        raise ValueError("Supporting passages exceed incoming budget")

    support_set = set(support_indices)
    remaining = np.asarray([i for i in range(len(candidates)) if i not in support_set], dtype=np.int64)
    sampled = rng.choice(remaining, size=FINAL - len(support_indices), replace=False)
    selected_doc_indices = np.concatenate([np.asarray(support_indices, dtype=np.int64), sampled])
    rng.shuffle(selected_doc_indices)
    selected = [candidates[int(i)] for i in selected_doc_indices]
    support_local = np.asarray([j for j, item in enumerate(selected) if item["support"]], dtype=np.int64)
    support_local_set = set(int(x) for x in support_local)
    incoming = support_local_set.copy()
    extra = [j for j in range(FINAL) if j not in incoming]
    incoming.update(int(x) for x in rng.choice(extra, size=20_000 - len(incoming), replace=False))
    incoming = np.asarray(sorted(incoming), dtype=np.int64)
    old = np.asarray([j for j in range(FINAL) if j not in set(incoming)], dtype=np.int64)
    if len(old) != BASE:
        raise AssertionError((len(old), BASE))

    encoder = SentenceTransformer(str(MODEL), local_files_only=True)
    doc_texts = [item["text"] for item in selected]
    vectors = encoder.encode(doc_texts, batch_size=128, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True,
                             device="cpu").astype("float32")
    queries = encoder.encode(selected_questions, batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True,
                             device="cpu").astype("float32")
    if vectors.shape != (FINAL, 384) or queries.shape != (N_QUERIES, 384):
        raise ValueError((vectors.shape, queries.shape))
    # These IDs index the frozen local pool.  Source IDs and question IDs are
    # retained separately so no benchmark identity is silently reused.
    source_ids = np.asarray(selected_doc_indices, dtype=np.int64)
    support_mask = np.zeros(FINAL, dtype=np.uint8)
    support_mask[support_local] = 1
    np.savez_compressed(DATA / "pool.npz", vectors=vectors, queries=queries,
                        source_ids=source_ids, query_source_ids=qids.astype(np.int64),
                        old_pool=old, new_pool=incoming,
                        new_query_pool=np.arange(N_QUERIES, dtype=np.int64),
                        support_mask=support_mask)
    question_records = [{"local_query": i, "source_row": int(qids[i]),
                         "question": selected_questions[i],
                         "supporting_titles": sorted({x[0] for x in supports_by_q[int(qids[i])]})}
                        for i in range(N_QUERIES)]
    (DATA / "queries.json").write_text(json.dumps(question_records, indent=2) + "\n")
    source_manifest = {"path": str(SOURCE), "sha256": sha(SOURCE), "bytes": SOURCE.stat().st_size,
                       "split": "distractor/validation", "rows": len(rows)}
    model_manifest = {"path": str(MODEL), "sha256": model_sha(MODEL), "license": "Apache-2.0",
                      "name": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384}
    manifest = dict(protocol="hotpot-semantic-order-v1", preparation_seed=SEED,
                    source=source_manifest, encoder=model_manifest,
                    final_documents=FINAL, initial_documents=BASE, update_rounds=5,
                    batch_size=4_000, query_count=N_QUERIES,
                    support_documents=int(len(support_local)),
                    pool_sha256=sha(out),
                    preprocessing="title-prefixed context sentences; encoder L2 normalization",
                    selection="512 validation questions; all their supporting passages plus fixed-seed distractors",
                    caveat="Text-document semantic validation, not an agent trajectory, RAG end-to-end answer benchmark, or production timestamp trace",
                    python=platform.python_version())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (DATA / "SOURCE.md").write_text(
        "# HotpotQA semantic fixture\n\n"
        "This fixture is derived from the public HotpotQA distractor validation split. "
        "Questions are queries and title-prefixed context sentences are documents. "
        "The local all-MiniLM-L6-v2 encoder produces 384-dimensional normalized vectors.\n\n"
        "The fixture is used only for dynamic vector-index protocol validation. It is not an agent trajectory, "
        "not a production update trace, and does not claim end-to-end QA accuracy. See the manifest for source and model hashes.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
