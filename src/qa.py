"""
qa.py
-----
The genuine "ask a question, get a RAG-grounded answer" feature: embed
the question, similarity-search the Chroma store that chunk_and_index
built, and ask the local LLM to answer using ONLY the retrieved chunks.

This is deliberately kept separate from the main graph: the diagram
pipeline needs a fixed, deterministic set of files traced (structural
analysis), while free-form questions ("what handles auth?") are exactly
the kind of "don't know which file in advance" problem RAG is built for.

Usage:
    python -m src.qa <vectorstore_dir> "What handles authentication?"
"""

import sys                                                    # to read command-line arguments

from langchain_chroma import Chroma                             # to re-open the persisted vector store
from .llm import get_embeddings, chat_complete                   # reuse the exact same wrappers as the main pipeline


def ask(vectorstore_dir: str, question: str, k: int = 5) -> str:
    # re-open the SAME persisted store the indexing step wrote to — this is
    # exactly why chunk_and_index_node stored a directory path in state
    # instead of a live, unpicklable Chroma object
    store = Chroma(persist_directory=vectorstore_dir, embedding_function=get_embeddings())

    # embeds `question` under the hood and returns the k most similar chunks by vector distance
    results = store.similarity_search(question, k=k)

    # stitch the retrieved chunks into one labeled block so both the model
    # and a human reader can trace any claim back to its source file
    context = "\n\n".join(
        f"# {doc.metadata.get('file_path', 'unknown')}\n{doc.page_content}"
        for doc in results
    )

    prompt = (
        "Answer the question using ONLY the code excerpts below. "
        "If the answer isn't in the excerpts, say so — don't guess.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    return chat_complete(prompt)


if __name__ == "__main__":                                      # only runs when invoked directly, not on import
    if len(sys.argv) < 3:
        print('Usage: python -m src.qa <vectorstore_dir> "question"')
        sys.exit(1)
    print(ask(sys.argv[1], sys.argv[2]))