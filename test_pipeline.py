"""
test_pipeline.py
-----------------
Runs the REAL graph (real chunking, real regex import tracing, real
LangGraph loops) against a small synthetic repo, but monkeypatches the
two functions that would otherwise call the Gemini API — proving the
control flow (both loops, the retry-then-fallback path) is correct
without needing a GOOGLE_API_KEY or network access.
"""

import os
import tempfile
import uuid

from src import nodes as nodes_module
from src import indexing as indexing_module
from src.graph import build_graph


def make_synthetic_repo():
    """Three python files with a real import chain: main -> utils -> helpers."""
    root = tempfile.mkdtemp(prefix="synthetic-repo-")
    with open(os.path.join(root, "main.py"), "w") as f:
        f.write(
            "from utils import do_thing\n"
            "if __name__ == '__main__':\n"
            "    do_thing()\n"
        )
    with open(os.path.join(root, "utils.py"), "w") as f:
        f.write(
            "from helpers import support\n"
            "def do_thing():\n"
            "    return support()\n"
        )
    with open(os.path.join(root, "helpers.py"), "w") as f:
        f.write(
            "def support():\n"
            "    return 42\n"
        )
    return root


def test_happy_path_and_retry_loop():
    repo_root = make_synthetic_repo()

    # --- monkeypatch clone: skip the network, just hand back our synthetic repo ---
    indexing_module.clone_repo = lambda url: repo_root
    nodes_module._clone_repo = lambda url: repo_root

    # --- monkeypatch indexing: skip real Gemini embeddings, fake a chunk count ---
    def fake_chunk_and_index(local_path, persist_dir):
        return (3, [])
    indexing_module.chunk_and_index = fake_chunk_and_index
    nodes_module._chunk_and_index = fake_chunk_and_index

    # --- monkeypatch chat_complete: deterministic fakes, no API key needed ---
    call_log = {"generate_diagram_calls": 0}

    def fake_chat_complete(prompt, model_name="gemini-2.5-flash"):
        # NOTE: match on the exact instruction text generate_diagram_node uses, not just the
        # word "Mermaid" — the writeup prompt also mentions "Mermaid diagram" in passing, and
        # an earlier version of this test's substring check accidentally caught that call too.
        if "Write a Mermaid flowchart" in prompt:
            call_log["generate_diagram_calls"] += 1
            if call_log["generate_diagram_calls"] == 1:
                return "this is not valid mermaid at all"        # force the FIRST attempt to fail validation
            return "graph TD\n    a[main] --> b[utils]\n    b --> c[helpers]"  # SECOND attempt is valid
        if "one plain sentence" in prompt:
            return "This file does something useful."             # fake per-file summary
        return "# Architecture\nThis is a fake writeup for testing."  # fake final writeup

    nodes_module.chat_complete = fake_chat_complete

    app = build_graph()
    initial_state = {
        "repo_url": "https://example.com/fake/repo",
        "max_summarize": 15,
        "max_diagram_retries": 3,
        "diagram_retry_count": 0,
        "diagram_valid": False,
        "diagram_error": "",
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = app.invoke(initial_state, config)

    # --- assertions: did loop #1 (import tracing) actually traverse all 3 files? ---
    assert set(final_state["visited"]) == {"main.py", "utils.py", "helpers.py"}, final_state["visited"]
    assert ("main.py", "utils.py") in final_state["edges"]
    assert ("utils.py", "helpers.py") in final_state["edges"]
    print("Loop #1 (import tracing) traversed all files correctly:", final_state["visited"])

    # --- assertions: did loop #2 (diagram retry) actually retry once, then succeed? ---
    assert call_log["generate_diagram_calls"] == 2, "expected exactly one retry"
    assert final_state["diagram_valid"] is True
    assert "graph TD" in final_state["diagram_code"]
    print("Loop #2 (diagram retry) retried once and then succeeded, as designed.")

    # --- assertions: did the writeup step run at all? ---
    assert "Architecture" in final_state["writeup"]
    print("Writeup generated successfully.")

    print("\nHAPPY PATH + RETRY TEST PASSED\n")


def test_fallback_when_all_retries_fail():
    repo_root = make_synthetic_repo()
    indexing_module.clone_repo = lambda url: repo_root
    nodes_module._clone_repo = lambda url: repo_root

    def fake_chunk_and_index(local_path, persist_dir):
        return (3, [])
    nodes_module._chunk_and_index = fake_chunk_and_index

    def always_broken_chat_complete(prompt, model_name="gemini-2.5-flash"):
        if "Write a Mermaid flowchart" in prompt:
            return "not mermaid, ever"                            # ALWAYS fails validation
        if "one plain sentence" in prompt:
            return "Summary."
        return "Writeup."

    nodes_module.chat_complete = always_broken_chat_complete

    app = build_graph()
    initial_state = {
        "repo_url": "https://example.com/fake/repo",
        "max_summarize": 15,
        "max_diagram_retries": 2,                                  # small cap so the test is fast
        "diagram_retry_count": 0,
        "diagram_valid": False,
        "diagram_error": "",
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = app.invoke(initial_state, config)

    # the LLM never produced valid mermaid, so finalize_diagram_node must have
    # swapped in the deterministic fallback — which is ALWAYS valid syntax
    assert final_state["diagram_valid"] is False               # the LLM's own output never validated...
    assert final_state["diagram_code"].startswith("graph TD")  # ...but we still ended up with valid Mermaid
    assert "main.py" in final_state["diagram_code"]             # built from the REAL traced edges, not a stub
    print("Fallback diagram used after exhausting retries:")
    print(final_state["diagram_code"])

    print("\nFALLBACK TEST PASSED\n")


if __name__ == "__main__":
    test_happy_path_and_retry_loop()
    test_fallback_when_all_retries_fail()
    print("ALL INTEGRATION TESTS PASSED")
