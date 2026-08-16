"""
main.py
-------
Codebase Cartographer — CLI entry point.

Usage:
    python main.py https://github.com/<user>/<repo>

Requires a GOOGLE_API_KEY in a .env file (see .env.example) — get a free
one at https://aistudio.google.com/apikey
"""

import sys                                                     # to read the repo URL from the command line
import uuid                                                     # to generate a unique thread_id per run
from dotenv import load_dotenv                                  # loads GOOGLE_API_KEY from .env into the environment

load_dotenv()                                                    # MUST run before importing src.graph, which
                                                                  # imports src.llm, which reads the API key at
                                                                  # client-construction time

from src.graph import build_graph                                # our compiled LangGraph app
from src.render import render_mermaid_to_png, mermaid_cli_available  # optional Mermaid -> PNG step


def main():
    if len(sys.argv) < 2:                                         # user forgot to pass a repo URL
        print("Usage: python main.py <github-repo-url>")
        sys.exit(1)

    repo_url = sys.argv[1]                                         # e.g. "https://github.com/psf/requests"
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

    with open("architecture.mmd", "w") as f:                         # save the diagram on its own, ready to paste
        f.write(final_state["diagram_code"])                          # into any Mermaid-aware renderer
    with open("architecture.md", "w") as f:                           # save the writeup on its own, ready for a README
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
