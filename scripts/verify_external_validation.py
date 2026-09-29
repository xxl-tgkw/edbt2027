#!/usr/bin/env python3
"""Verify the compact external scale and rebuild-frequency evidence ledgers."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def verify_scale(folder, size, seeds):
    env = read(folder + "/environment.json")
    rows = read(folder + "/raw.json")
    summary = read(folder + "/summary.json")
    if env["protocol"] != "semantic-scale-v1" or len(rows) != len(seeds) * 6 or len(summary) != 14:
        raise ValueError(f"invalid scale ledger: {folder}")
    expected = {(s, c, p) for s in seeds for c in ["shifted", "shuffled"]
                for p in ["flat", "ivf_fixed", "ivf_rebuilt"]}
    got = {(r["seed"], r["condition"], r["policy"]) for r in rows}
    if got != expected:
        raise ValueError(f"incomplete scale matrix: {folder}")
    for row in rows:
        if row["size"] != size or len(row["events"]) != 6:
            raise ValueError("wrong scale size or rounds")
        if set(row["summaries"]) != {"0"} if row["policy"] == "flat" else set(row["summaries"]) != {"4", "16", "64"}:
            raise ValueError("wrong scale budgets")
        for value in row["summaries"].values():
            if not 0 <= value["recall"] <= 1 or not np.isfinite(value["service_ms"]):
                raise ValueError("invalid scale summary")
    return {"folder": folder, "rows": len(rows), "summary_rows": len(summary), "size": size}


def verify_schedule(folder):
    env = read(folder + "/environment.json")
    rows = read(folder + "/raw.json")
    summary = read(folder + "/summary.json")
    if env["protocol"] != "semantic-rebuild-sensitivity-v1" or len(rows) != 40 or len(summary) != 16:
        raise ValueError("invalid rebuild sensitivity ledger")
    expected = {(s, c, schedule) for s in range(5) for c in ["shifted", "shuffled"]
                for schedule in ["fixed", "period2", "period1", "period4"]}
    got = {(r["seed"], r["condition"], r["schedule"]) for r in rows}
    if got != expected:
        raise ValueError("incomplete rebuild sensitivity matrix")
    for row in rows:
        if len(row["events"]) != 6 or set(row["summaries"]) != {"16", "64"}:
            raise ValueError("invalid rebuild sensitivity row")
        for value in row["summaries"].values():
            if not 0 <= value["recall"] <= 1 or not np.isfinite(value["service_ms"]):
                raise ValueError("invalid rebuild sensitivity summary")
    return {"folder": folder, "rows": len(rows), "summary_rows": len(summary)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale100", default="results/semantic_scale_100k_v2")
    parser.add_argument("--scale250", default="results/semantic_scale_250k_v2")
    parser.add_argument("--schedule", default="results/semantic_rebuild_sensitivity_v1")
    args = parser.parse_args()
    report = [verify_scale(args.scale100, 100000, [0, 1, 2]), verify_scale(args.scale250, 250000, [0, 1, 2]), verify_schedule(args.schedule)]
    output = ROOT / "results/external_validation_verification.json"
    output.write_text(json.dumps({"status": "passed", "checks": report}, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checks": report}, indent=2))


if __name__ == "__main__":
    main()
