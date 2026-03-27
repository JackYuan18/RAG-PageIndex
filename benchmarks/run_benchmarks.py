#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_subprocess(cmd: list[str]) -> tuple[int, float]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return p.returncode, round(time.perf_counter() - t0, 3)


def load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def indexing_meta_entry(pdf_arg: str, timings_path: str, wall_seconds: float) -> dict:
    data = load_json(timings_path) or {}
    pdf_path = data.get("doc") or pdf_arg
    steps = data.get("step_timings") or []
    return {
        "pdf_path": pdf_path,
        "wall_seconds": wall_seconds,
        "step_timings": steps,
    }


def rag_meta_entry(query_arg: str, out_path: str, wall_seconds: float) -> dict:
    data = load_json(out_path) or {}
    q = data.get("query") or query_arg
    steps = data.get("step_timings") or []
    return {
        "query": q,
        "wall_seconds": wall_seconds,
        "step_timings": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "benchmarks", "runs"))
    ap.add_argument(
        "--pdf",
        action="append",
        default=["/home/zyuan/NSTSCE_Bot/PageIndex/Database/NSTSCE_L3System_Final.pdf",
        "/home/zyuan/NSTSCE_Bot/PageIndex/Database/AAA NSTSCE Training Drivers on L2 Systems.pdf",
        "/home/zyuan/NSTSCE_Bot/PageIndex/Database/AVSC Best Practice for Developing ADS.pdf",
        "/home/zyuan/NSTSCE_Bot/PageIndex/Database/NSTSCE_NewTechOutreach_Final.pdf"],
    )
    ap.add_argument(
        "--query",
        action="append",
        default=["In vtti, who works on the challenges of L3 autonomous systems?",
        "What are the challenges of L3 autonomous driving systems?",
        "Who works on Gaussian process regression?"],
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    results = {"run_id": run_id, "model": args.model, "indexing": [], "rag": []}

    total_jobs = len(args.pdf) + len(args.query)
    done = 0

    for pdf in args.pdf:
        done += 1
        base = os.path.splitext(os.path.basename(pdf))[0]
        timings_out = os.path.join(run_dir, f"index_{base}.json")
        cmd = [
            "python3",
            "run_pageindex.py",
            "--pdf_path",
            pdf,
            "--model",
            args.model,
            #"--update_docindex",
            "--timings-out",
            timings_out,
        ]
        print(f"[{done}/{total_jobs}] Indexing: {pdf}")
        rc, wall = run_subprocess(cmd)
        results["indexing"].append(indexing_meta_entry(pdf, timings_out, wall))
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"[{done}/{total_jobs}] Indexing done: {base} -> {status} in {wall:.2f}s")

    for i, q in enumerate(args.query):
        done += 1
        out = os.path.join(run_dir, f"rag_{i+1}.json")
        cmd = ["python3", "RAG/rag_query.py", "--query", q, "--model", args.model, "--out", out]
        short_q = q if len(q) <= 80 else q[:77] + "..."
        print(f"[{done}/{total_jobs}] RAG query: {short_q}")
        rc, wall = run_subprocess(cmd)
        results["rag"].append(rag_meta_entry(q, out, wall))
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"[{done}/{total_jobs}] RAG done: {status} in {wall:.2f}s")

    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Benchmark artifacts saved to: {run_dir}")
    print(run_dir)


if __name__ == "__main__":
    main()
