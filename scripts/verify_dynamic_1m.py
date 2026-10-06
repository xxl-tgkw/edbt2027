#!/usr/bin/env python3
"""Audit the 1M dynamic-index event log without rerunning timings."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    root = Path(ap.parse_args().input)
    rows = json.loads((root / "raw.json").read_text())
    assert len(rows) == 6, len(rows)
    workloads = {}
    for path in root.glob("workload_*.json"):
        meta = json.loads(path.read_text()); workloads[meta["condition"]] = meta
    assert set(workloads) == {"shifted", "shuffled"}
    assert workloads["shifted"]["base_membership_sha256"] == workloads["shuffled"]["base_membership_sha256"]
    assert workloads["shifted"]["update_membership_sha256"] == workloads["shuffled"]["update_membership_sha256"]
    assert workloads["shifted"]["source_ids"][:800000] == workloads["shuffled"]["source_ids"][:800000]
    assert workloads["shifted"]["source_ids"][800000:] != workloads["shuffled"]["source_ids"][800000:]
    seen = set()
    for row in rows:
        key = (row["seed"], row["condition"], row["policy"]); assert key not in seen; seen.add(key)
        assert row["condition"] in {"shifted", "shuffled"}
        assert row["policy"] in {"ivf_fixed", "ivf_rebuilt", "hnsw"}
        assert len(row["events"]) == 5
        assert row["corpus_sha256"] == json.loads((root / f"workload_{row['condition']}_{row['seed']}.json").read_text())["corpus_sha256"]
        for event in row["events"]:
            assert event["indexed"] in {840000, 880000, 920000, 960000, 1000000}
            for result in event["budgets"].values():
                for name in ("validation_recall", "test_recall", "query_p95_ms", "query_mean_ms"):
                    assert np.isfinite(result[name]), (key, name)
    report = {"protocol": "semantic-1m-dynamic-v1", "rows": len(rows),
              "events": sum(len(r["events"]) for r in rows), "unique_jobs": len(seen),
              "status": "pass"}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
