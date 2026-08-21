"""
run_eval.py
-----------
Runs the pipeline's structural analysis (entry-point detection + import
tracing — the deterministic, no-LLM half of the pipeline; see README's
"Where RAG is used" section for why this half is deliberately NOT RAG)
against every repo in eval/dataset.py, scores it against hand-labeled
ground truth, and prints a report.

Deliberately does NOT touch the LLM-dependent half (summaries, diagram,
writeup) — there's no ground truth for "the correct summary of a file",
and scoring free-text against free-text needs a different kind of
evaluation (e.g. an LLM-as-judge rubric) that's a reasonable next step
but a separate piece of work from this.

Usage:
    python -m eval.run_eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so `src` imports work when run directly

from src.indexing import clone_repo, list_source_files
from src import nodes as n
from eval.dataset import EVAL_REPOS
from eval.metrics import precision_recall_f1


def run_structural_pipeline(local_path: str) -> dict:
    """Runs exactly the same non-LLM nodes the real graph runs, in the same order —
    identify_entry_points_node, then trace_step_node looped to completion."""
    all_files = list_source_files(local_path)
    state = {"local_path": local_path, "all_files": all_files}
    state.update(n.identify_entry_points_node(state))
    while state["frontier"]:                          # the exact same loop condition as frontier_router
        state.update(n.trace_step_node(state))
    return state


def evaluate_repo(entry: dict) -> dict:
    repo_url = entry["repo_url"]
    if os.path.isdir(repo_url):                          # same check clone_repo_node uses — a local fixture,
        print(f"Using local folder {repo_url} ...")        # not a URL, so skip git entirely
        local_path = repo_url
    else:
        print(f"Cloning {repo_url} ...")
        local_path = clone_repo(repo_url)                   # a real, fresh git clone every run — same as production

    state = run_structural_pipeline(local_path)
    predicted_entries = set(state["entry_points"])
    predicted_edges = set(state["edges"])

    return {
        "name": entry["name"],
        "entry_points": precision_recall_f1(predicted_entries, entry["true_entry_points"]),
        "edges": precision_recall_f1(predicted_edges, entry["true_edges"]),
    }


def print_report(results: list):
    print("\n" + "=" * 72)
    print("EVALUATION REPORT — structural analysis (entry points + import tracing)")
    print("=" * 72)

    for r in results:
        ep, eg = r["entry_points"], r["edges"]
        print(f"\n{r['name']}")
        print(f"  Entry points  —  precision: {ep['precision']:.0%}  recall: {ep['recall']:.0%}  f1: {ep['f1']:.0%}")
        if ep["fn"]:
            print(f"    missed entry points (false negatives): {ep['false_negatives']}")
        if ep["fp"]:
            print(f"    wrong entry points (false positives):  {ep['false_positives']}")
        print(f"  Import edges  —  precision: {eg['precision']:.0%}  recall: {eg['recall']:.0%}  f1: {eg['f1']:.0%}")
        if eg["fn"]:
            print(f"    missed edges (false negatives): {eg['false_negatives']}")
        if eg["fp"]:
            print(f"    wrong edges (false positives):  {eg['false_positives']}")

    # macro-average: every repo counts equally, regardless of its size —
    # avoids one large repo dominating the score
    n_repos = len(results)
    avg_entry_f1 = sum(r["entry_points"]["f1"] for r in results) / n_repos
    avg_edge_f1 = sum(r["edges"]["f1"] for r in results) / n_repos

    print("\n" + "-" * 72)
    print(f"OVERALL (macro-average across {n_repos} repos)")
    print(f"  Entry-point detection F1: {avg_entry_f1:.0%}")
    print(f"  Import-edge detection F1: {avg_edge_f1:.0%}")
    print("=" * 72)


def main():
    results = [evaluate_repo(entry) for entry in EVAL_REPOS]
    print_report(results)


if __name__ == "__main__":
    main()