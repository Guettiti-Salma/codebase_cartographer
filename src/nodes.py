"""
nodes.py
--------
Every node function in the graph. Each one takes the current RepoState
dict, does one job, and returns ONLY the fields it changed — LangGraph
merges that partial dict back into the full state automatically, so a
node never has to know about (or repeat) fields it doesn't touch.
"""

import os     # path joins, for reading files back off disk
import re     # regex checks used by the Mermaid validator
import tempfile  # a fresh directory for this run's Chroma index

from .state import RepoState                                    # the shape every node reads/writes
from .indexing import clone_repo as _clone_repo                  # renamed on import: nodes.py owns the "_node" names
from .indexing import chunk_and_index as _chunk_and_index
from .indexing import list_source_files
from .parsing import is_entry_point, extract_imports, resolve_import
from .llm import chat_complete


# ============================================================
# Straight-line setup steps (no looping, no branching)
# ============================================================

def clone_repo_node(state: RepoState) -> dict:
    repo_url = state["repo_url"]
    if os.path.isdir(repo_url):                                    # a local folder was passed instead of a URL —
        return {"local_path": repo_url}                             # use it directly, skip git entirely. This is
                                                                      # what lets examples/tiny_repo (or any local
                                                                      # project) be analyzed with zero network calls
                                                                      # and the smallest possible embedding volume.
    local_path = _clone_repo(repo_url)                               # otherwise, shell out to `git clone` as normal
    return {"local_path": local_path}                                # only report what this node changed


def chunk_and_index_node(state: RepoState) -> dict:
    persist_dir = tempfile.mkdtemp(prefix="repo-architect-index-")   # fresh Chroma directory for this run
    count, _docs = _chunk_and_index(state["local_path"], persist_dir)  # the real RAG indexing step
    all_files = list_source_files(state["local_path"])              # repo-relative paths, reused by later nodes
    return {
        "vectorstore_dir": persist_dir,
        "indexed_file_count": count,
        "all_files": all_files,
    }


def identify_entry_points_node(state: RepoState) -> dict:
    entries = [
        f for f in state["all_files"]
        if is_entry_point(f, state["local_path"])                   # structural check — no LLM call needed
    ]
    if not entries:                                                  # nothing matched a known convention
        entries = state["all_files"][:1]                             # fall back to "just start somewhere"
    return {
        "entry_points": entries,
        "frontier": list(entries),                                  # copy — trace_step_node will mutate its own list
        "visited": [],
        "edges": [],
    }


# ============================================================
# Loop #1 — keep tracing imports until the frontier is empty
# ============================================================

def trace_step_node(state: RepoState) -> dict:
    frontier = list(state["frontier"])                               # copy: never mutate the incoming state in place
    current = frontier.pop(0)                                        # process exactly ONE file per node call
    visited = state["visited"] + [current]                           # this file is now done
    edges = list(state["edges"])                                     # copy, we'll append new dependency edges to it

    full_path = os.path.join(state["local_path"], current)           # absolute path to actually read
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        content = ""                                                  # unreadable — treat as "no imports found"

    all_files_set = set(state["all_files"])                           # set for fast membership checks below
    for raw_import in extract_imports(current, content):               # every import statement in this one file
        resolved = resolve_import(current, raw_import, state["local_path"], all_files_set)
        if resolved is None:
            continue                                                   # third-party/stdlib import — not our concern
        if (current, resolved) not in edges:                            # a file can import the same dependency
            edges.append((current, resolved))                           # via multiple statements — record it once
        if resolved not in visited and resolved not in frontier:       # avoid duplicate work and infinite loops
            frontier.append(resolved)                                  # queue it for a FUTURE trace_step call

    return {"frontier": frontier, "visited": visited, "edges": edges}


def frontier_router(state: RepoState) -> str:
    return "loop" if state["frontier"] else "done"                    # the entire loop condition, in one line


# ============================================================
# Summarization — one LLM call per traced file, capped for cost
# ============================================================

def summarize_modules_node(state: RepoState) -> dict:
    summaries = {}
    files_to_summarize = state["visited"][: state["max_summarize"]]     # cap how many files call the LLM
    for file_path in files_to_summarize:
        full_path = os.path.join(state["local_path"], file_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                snippet = f.read()[:1500]                                # first ~1500 chars is plenty for a summary
        except OSError:
            continue
        prompt = f"In one plain sentence, what does this file do?\n\n{snippet}"
        summaries[file_path] = chat_complete(prompt)     # uses the default local model (see llm.py)
    return {"module_summaries": summaries}


# ============================================================
# Loop #2 — generate a diagram, validate it, retry on failure
# ============================================================

def _format_edges(edges) -> str:
    return "\n".join(f"{src} -> {dst}" for src, dst in edges) or "(no dependencies traced)"


def generate_diagram_node(state: RepoState) -> dict:
    edge_text = _format_edges(state["edges"])
    summary_text = "\n".join(f"{f}: {s}" for f, s in state["module_summaries"].items())

    retry_note = ""
    if state.get("diagram_error"):                                       # only set on a retry attempt
        retry_note = (
            f"\nYour previous attempt was invalid: {state['diagram_error']}\n"
            f"Previous attempt:\n{state['diagram_code']}\nFix it and try again.\n"
        )

    prompt = (
        "Write a Mermaid flowchart (start with 'graph TD') showing these file "
        "dependencies. Use short labels, not full paths, inside the brackets. "
        "Output ONLY the Mermaid code — no markdown fences, no explanation.\n\n"
        f"Dependencies:\n{edge_text}\n\nFile summaries:\n{summary_text}\n{retry_note}"
    )
    diagram = chat_complete(prompt)
    return {
        "diagram_code": diagram,
        "diagram_retry_count": state["diagram_retry_count"] + 1,          # count this attempt toward the cap
    }


def _validate_mermaid(code: str):
    cleaned = code.strip()
    # the model sometimes wraps output in ```mermaid fences despite being told not to — strip them before checking
    cleaned = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    if not re.match(r"^(graph|flowchart)\s+(TD|TB|LR|RL|BT)", cleaned):
        return False, "Diagram must start with 'graph TD' or 'flowchart LR' (or similar direction)."
    if "-->" not in cleaned and "---" not in cleaned:
        return False, "Diagram has no edges (no '-->' or '---' found)."
    if cleaned.count("[") != cleaned.count("]"):
        return False, "Unbalanced [ ] brackets in node labels."
    if cleaned.count("(") != cleaned.count(")"):
        return False, "Unbalanced ( ) parentheses in node labels."
    return True, ""


def validate_diagram_node(state: RepoState) -> dict:
    valid, error = _validate_mermaid(state["diagram_code"])               # deterministic check, no LLM call
    return {"diagram_valid": valid, "diagram_error": error}


def diagram_router(state: RepoState) -> str:
    if state["diagram_valid"]:
        return "proceed"                                                   # success — move on to the writeup
    if state["diagram_retry_count"] >= state["max_diagram_retries"]:
        return "proceed"                                                   # give up gracefully, don't loop forever
    return "retry"                                                         # try again, with the error as feedback


def _fallback_diagram(edges) -> str:
    """A deterministic, always-valid diagram built directly from the edge
    list — used only if the LLM never produces valid Mermaid syntax within
    the retry budget. Guarantees the pipeline still finishes with SOMETHING
    useful instead of erroring out."""
    lines = ["graph TD"]
    seen_ids = {}

    def node_id(path: str) -> str:
        # Mermaid node IDs can't contain slashes or dots — build a safe short alias per file
        if path not in seen_ids:
            seen_ids[path] = f"n{len(seen_ids)}"
        return seen_ids[path]

    for src, dst in edges:
        lines.append(f'    {node_id(src)}["{src}"] --> {node_id(dst)}["{dst}"]')
    if not edges:
        lines.append('    n0["No dependencies traced"]')
    return "\n".join(lines)


def finalize_diagram_node(state: RepoState) -> dict:
    if state["diagram_valid"]:
        # strip any stray code fences even on a valid result, so the saved .mmd file is clean
        cleaned = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", state["diagram_code"].strip(), flags=re.MULTILINE)
        return {"diagram_code": cleaned.strip()}
    # every retry failed validation — fall back to a diagram we KNOW is syntactically valid
    return {"diagram_code": _fallback_diagram(state["edges"])}


# ============================================================
# Final write-up
# ============================================================

def generate_writeup_node(state: RepoState) -> dict:
    summary_text = "\n".join(f"- **{f}**: {s}" for f, s in state["module_summaries"].items())
    prompt = (
        f"Write a short markdown architecture overview for the repo at {state['repo_url']}.\n"
        f"Entry points: {', '.join(state['entry_points'])}\n"
        f"Module summaries:\n{summary_text}\n\n"
        f"A Mermaid diagram of the dependencies has already been generated separately — "
        f"don't repeat it, just refer to it.\n\n"
        "Explain what the project does, how the main pieces connect, and where a new "
        "contributor should start reading."
    )
    writeup = chat_complete(prompt)
    return {"writeup": writeup}