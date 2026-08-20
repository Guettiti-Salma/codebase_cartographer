"""
main.py
-------
Codebase Cartographer — CLI entry point.

Usage:
    python main.py https://github.com/<user>/<repo>
    python main.py examples/tiny_repo          # or any local folder — skips git

Runs fully locally — no API key needed. Requires:
  - Ollama running locally with the chat model pulled: `ollama pull qwen2.5-coder:3b`
  - (embeddings download automatically on first use — see src/llm.py)
"""

import os                                                       # path normalization for local-folder input
import sys                                                     # to read the repo URL from the command line
import uuid                                                     # to generate a unique thread_id per run

from src.graph import build_graph                                # our compiled LangGraph app
from src.render import render_mermaid_to_png, mermaid_cli_available  # optional Mermaid -> PNG step


def main():
    if len(sys.argv) < 2:                                         # user forgot to pass a repo URL
        print("Usage: python main.py <github-repo-url-or-local-folder>")
        sys.exit(1)

    # normalize the raw argument before anything else touches it:
    #  - .strip() removes trailing newlines/spaces from a pasted path
    #  - .strip('"\'') removes wrapping quotes — Windows Explorer's "Copy as path"
    #    wraps paths in double quotes, which would otherwise make os.path.isdir()
    #    return False even for a perfectly real folder
    #  - os.path.expanduser() turns "~/projects/foo" into an actual absolute path
    repo_url = os.path.expanduser(sys.argv[1].strip().strip('"').strip("'"))

    # fail fast with a clear message if this is neither a real local folder nor
    # something that looks like a git URL — better than a cryptic git error later
    looks_like_url = repo_url.startswith(("http://", "https://", "git@"))
    if not os.path.isdir(repo_url) and not looks_like_url:
        print(f"'{repo_url}' isn't a folder that exists on this machine, and it")
        print("doesn't look like a git URL (expected it to start with https:// or git@).")
        print("Usage: python main.py <github-repo-url-or-local-folder>")
        sys.exit(1)

    app = build_graph()                                             # compile the graph once, reuse for this run

    print("Codebase Cartographer")                                  # small banner — purely cosmetic
    print("=" * 22)

    initial_state = {                                                # every field the FIRST node needs to exist
        "repo_url": repo_url,
        "max_summarize": 15,                                         # cap LLM summary calls — cost/time control
        "max_diagram_retries": 3,                                    # cap self-correction attempts
        "diagram_retry_count": 0,
        "diagram_valid": False,
        "diagram_error": "",
    }

    # thread_id lets the checkpointer track this run distinctly from any other run
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Analyzing {repo_url} ...")
    final_state = app.invoke(initial_state, config)                  # runs the WHOLE graph, including both loops

    print("\n--- Mermaid diagram ---\n")
    print(final_state["diagram_code"])

    print("\n--- Writeup ---\n")
    print(final_state["writeup"])

    with open("architecture.mmd", "w", encoding="utf-8") as f:        # save the diagram on its own, ready to paste
        f.write(final_state["diagram_code"])                          # into any Mermaid-aware renderer
    with open("architecture.md", "w", encoding="utf-8") as f:          # save the writeup on its own, ready for a README
        f.write(final_state["writeup"])

    print("\nSaved architecture.mmd and architecture.md")

    # --- optional: render the .mmd straight to a real .png, if mermaid-cli is installed ---
    if mermaid_cli_available():
        rendered = render_mermaid_to_png("architecture.mmd", "architecture.png")
        if rendered:
            print("Saved architecture.png")
        else:
            # mmdc IS installed but the render still failed — usually means the headless
            # browser it depends on isn't set up; the text/writeup outputs are unaffected
            print("Could not render architecture.png (mmdc is installed but rendering failed).")
            print("The .mmd and .md files are still saved and complete.")
    else:
        print("Skipped architecture.png — mermaid-cli isn't installed.")
        print("Install it with: npm install -g @mermaid-js/mermaid-cli")
        print("Or paste architecture.mmd into https://mermaid.live to view it without installing anything.")

    print(f"Vector store persisted at: {final_state['vectorstore_dir']}")
    print('Ask follow-up questions with:')
    print(f'  python -m src.qa "{final_state["vectorstore_dir"]}" "your question here"')


if __name__ == "__main__":                                           # only runs main() when executed directly
    main()