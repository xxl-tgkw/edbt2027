#!/usr/bin/env python3
"""Rebuild-frequency sensitivity for the fixed/rebuilt IVF policies."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from prepare_semantic_data import sha
from run_semantic import NativeIndex, load_pool, make_workload

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "semantic-rebuild-sensitivity-v1"
SCHEDULES = {"fixed": (), "period2": (2, 4), "period1": (1, 2, 3, 4, 5), "period4": (4,)}
BUDGETS = (16, 64)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def evaluate(x, q, truth, meta, schedule_name, condition, seed):
    schedule = set(SCHEDULES[schedule_name])
    start = time.perf_counter_ns()
    index = NativeIndex(x[:meta["base"]], "ivf_rebuilt" if schedule else "ivf_fixed", seed)
    build_ms = (time.perf_counter_ns() - start) / 1e6
    for z in x[:10]:
        index.index.search(z[None], 10)
    events = []
    maintenance = 0.0
    for round_id in range(meta["rounds"] + 1):
        insert_ms = rebuild_ms = 0.0
        if round_id:
            lo = meta["base"] + (round_id - 1) * meta["batch"]
            hi = meta["base"] + round_id * meta["batch"]
            start = time.perf_counter_ns()
            index.index.add(x[lo:hi])
            insert_ms = (time.perf_counter_ns() - start) / 1e6
        rebuilt = round_id in schedule
        if rebuilt:
            start = time.perf_counter_ns()
            index.rebuild(x[:meta["base"] + round_id * meta["batch"]])
            rebuild_ms = (time.perf_counter_ns() - start) / 1e6
        budgets = {}
        for budget in BUDGETS:
            index.index.nprobe = budget
            result = []
            for query, nn in zip(q[round_id, 1], truth[round_id][1]):
                _, returned = index.index.search(query.reshape(1, -1), 10)
                result.append(len(set(returned[0].tolist()) & set(nn)) / 10)
            budgets[str(budget)] = {"recall": float(np.mean(result))}
        maintenance += insert_ms + rebuild_ms
        events.append({"round": round_id, "rebuilt": rebuilt, "insert_ms": insert_ms,
                       "rebuild_ms": rebuild_ms, "budgets": budgets})
    # Retiming query work separately avoids using one budget's duration as a proxy for another.
    for budget in BUDGETS:
        index = NativeIndex(x[:meta["base"]], "ivf_rebuilt" if schedule else "ivf_fixed", seed)
        for round_id in range(1, meta["rounds"] + 1):
            lo = meta["base"] + (round_id - 1) * meta["batch"]
            hi = meta["base"] + round_id * meta["batch"]
            index.index.add(x[lo:hi])
            if round_id in schedule:
                index.rebuild(x[:hi])
            index.index.nprobe = budget
            times = []
            for query in q[round_id, 1]:
                start = time.perf_counter_ns()
                index.index.search(query.reshape(1, -1), 10)
                times.append((time.perf_counter_ns() - start) / 1e6)
            events[round_id]["budgets"][str(budget)]["query_mean_ms"] = float(np.mean(times))
            events[round_id]["budgets"][str(budget)]["query_p95_ms"] = float(np.percentile(times, 95))
    summaries = {}
    for budget in BUDGETS:
        rows = [event["budgets"][str(budget)] for event in events[1:]]
        summaries[str(budget)] = {
            "recall": float(np.mean([r["recall"] for r in rows])),
            "query_mean_ms": float(np.mean([r["query_mean_ms"] for r in rows])),
            "query_p95_ms": float(np.mean([r["query_p95_ms"] for r in rows])),
            "service_ms": maintenance + float(np.sum([r["query_mean_ms"] * meta["nq"] for r in rows])),
        }
    return {"protocol": PROTOCOL, "seed": seed, "condition": condition,
            "schedule": schedule_name, "build_ms": build_ms,
            "maintenance_ms": maintenance, "events": events, "summaries": summaries,
            "corpus_sha256": meta["corpus_sha256"], "queries_sha256": meta["queries_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True)
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        pool, manifest = load_pool()
        env = {"protocol": PROTOCOL, "args": vars(args), "python": platform.python_version(),
               "faiss": faiss.__version__, "source_manifest": manifest,
               "source_hashes": {"run_semantic.py": sha(ROOT / "scripts/run_semantic.py"),
                                 "prepare_semantic_data.py": sha(ROOT / "scripts/prepare_semantic_data.py")}}
        rows = []
        for seed in args.seeds:
            for condition in ("shifted", "shuffled"):
                x, q, truth, meta = make_workload(pool, seed, condition)
                for schedule in SCHEDULES:
                    row = evaluate(x, q, truth, meta, schedule, condition, seed)
                    rows.append(row)
                    print(f"{len(rows)}/{len(args.seeds)*len(SCHEDULES)*2} seed={seed} {condition} {schedule}", flush=True)
        (out / "raw.json").write_text(json.dumps(rows) + "\n")
        env["rows"] = len(rows)
        (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    summary = []
    for condition in ("shifted", "shuffled"):
        for schedule in SCHEDULES:
            group = [r for r in rows if r["condition"] == condition and r["schedule"] == schedule]
            for budget in BUDGETS:
                values = [r["summaries"][str(budget)] for r in group]
                summary.append({"condition": condition, "schedule": schedule, "budget": budget,
                                "seeds": len(group),
                                **{key: float(np.mean([v[key] for v in values])) for key in values[0]}})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Rebuild-frequency sensitivity", "", "Five seeds per condition and schedule; this is separate from the primary ten-seed study.",
             "", "| Order | Schedule | Budget | Recall | p95 ms | Service ms |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['schedule']} | {row['budget']} | {row['recall']:.4f} | {row['query_p95_ms']:.3f} | {row['service_ms']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
