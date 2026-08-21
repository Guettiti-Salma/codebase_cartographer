"""
dataset.py
----------
Hand-labeled ground truth for evaluating the pipeline's structural
analysis (entry-point detection + import tracing) against real repos.

Every true_entry_points / true_edges value below was verified by directly
reading the repo's source — not by trusting the pipeline's own output,
which would be circular and prove nothing.

Correction, made honestly rather than quietly: an earlier version of this
file claimed dbader/schedule's test_schedule.py has an `if __name__ ==
"__main__":` guard. That was wrong — a direct `grep -rn "__main__"` across
the whole repo returns nothing. No file in that repo matches this tool's
entry-point heuristics at all; it's a pure library with no conventional
entry point, which makes it a genuinely bad fit for this specific metric,
not a case where the tool is "wrong." It's been dropped from entry-point
scoring for that reason (see below), rather than leaving an incorrect
ground-truth label in place to make a number look better.
"""

EVAL_REPOS = [
    {
        "name": "pypa/sampleproject",
        "repo_url": "https://github.com/pypa/sampleproject",
        # verified by reading tests/test_simple.py directly
        "true_entry_points": {"tests/test_simple.py"},
        # src-layout: the real file lives under src/, not at the repo root —
        # this repo is exactly what caught the original resolve_import bug
        "true_edges": {("tests/test_simple.py", "src/sample/simple.py")},
    },
    {
        "name": "examples/tiny_repo (local fixture)",
        # a local folder, not a URL — run_eval.py detects this the same way
        # clone_repo_node does (os.path.isdir) and skips git entirely
        "repo_url": "examples/tiny_repo",
        # unambiguous by construction: main.py both matches ENTRY_POINT_FILENAMES
        # AND has a real __main__ guard — no judgment call involved in this label
        "true_entry_points": {"main.py"},
        "true_edges": {("main.py", "helper.py")},
    },
]

# --- Adding a repo ---
# 1. Pick something small (under ~15 traceable files) so hand-verification stays honest.
# 2. Clone it and actually read the files — don't run the pipeline first and copy its
#    answer, that's circular. Use `grep -rn "__main__"` across the whole repo, not just
#    the file you assume is the entry point — the schedule mistake above happened
#    specifically from assuming instead of checking every file.
# 3. true_entry_points: repo-relative paths of files with a `__main__` guard or a
#    conventional entry filename (main.py, app.py, index.js, ...). If NOTHING in the
#    repo qualifies (a pure library, for instance), that repo is a bad fit for
#    entry-point scoring specifically — don't force a label, pick a different repo.
# 4. true_edges: (importer_path, imported_path) pairs for every import that resolves
#    to another file IN this repo. Skip stdlib/third-party imports — resolve_import
#    is only meant to trace internal dependencies.
# 5. Append a new dict here with the same shape as the entries above.