"""
state.py
---------
Defines the single shared "state" object that flows through every node
in the LangGraph graph. Every node reads from this dict and returns a
partial update that LangGraph merges back in automatically — a node
never has to know about fields it doesn't use.
"""

from typing import TypedDict, List, Tuple, Dict  # TypedDict lets us type-hint a plain dict's shape


class RepoState(TypedDict):
    # --- inputs, set once before the graph starts running ---
    repo_url: str                       # the GitHub URL the user gave us
    max_summarize: int                  # cap on how many files we summarize with the LLM (cost control)
    max_diagram_retries: int            # cap on how many times we let the LLM retry a broken diagram

    # --- set by clone_repo_node ---
    local_path: str                     # where the repo was cloned to on disk

    # --- set by chunk_and_index_node ---
    vectorstore_dir: str                 # path to the persisted Chroma DB (we store the PATH, not the live
                                          # object, so the state stays serializable for LangGraph's checkpointer)
    indexed_file_count: int              # how many chunks got embedded, just for reporting to the user
    all_files: List[str]                 # every repo-relative source file path we found, reused by later nodes

    # --- set by identify_entry_points_node ---
    entry_points: List[str]              # repo-relative paths that look like where execution starts

    # --- used and updated by trace_step_node (loop #1: the import-tracing loop) ---
    frontier: List[str]                  # files we still need to trace imports for
    visited: List[str]                   # files we've already traced (avoids re-processing / infinite loops)
    edges: List[Tuple[str, str]]         # (importer_path, imported_path) pairs — the dependency graph itself

    # --- set by summarize_modules_node ---
    module_summaries: Dict[str, str]     # file path -> one-line, LLM-written summary of what it does

    # --- used and updated by generate_diagram_node / validate_diagram_node (loop #2: the self-correction loop) ---
    diagram_code: str                    # the latest Mermaid syntax the LLM produced
    diagram_valid: bool                  # did the last attempt pass validation?
    diagram_error: str                   # if invalid, what the validator said was wrong (fed back to the LLM)
    diagram_retry_count: int             # how many attempts we've made so far

    # --- set by generate_writeup_node ---
    writeup: str                         # the final markdown explanation of the architecture
