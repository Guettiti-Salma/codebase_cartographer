"""
indexing.py
-----------
Everything involved in turning a cloned repo into a searchable vector
store: walking the file tree, chunking each file with a language-aware
splitter, embedding the chunks with Gemini, and persisting them to a
local Chroma database. This IS the "RAG" part of the project — chunks
+ embeddings + vector database, exactly as advertised.
"""

import os          # filesystem walking and path joins
import subprocess  # to shell out to `git clone`
import tempfile    # to get a fresh scratch directory for the clone

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language  # language-aware chunking
from langchain_core.documents import Document   # the (text + metadata) unit LangChain passes around
from langchain_chroma import Chroma              # the vector store wrapper

from .parsing import SKIP_DIRS, detect_language  # reuse the same language/skip rules everywhere
from .llm import get_embeddings                  # our Gemini embeddings wrapper


def clone_repo(repo_url: str) -> str:
    target_dir = tempfile.mkdtemp(prefix="repo-architect-")               # fresh, unique scratch directory
    # --depth 1: only the latest snapshot, not the whole git history — much faster for a big repo
    subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
    return target_dir                                                     # hand back where it landed on disk


# Maps our simple language tags to LangChain's Language enum, which knows
# how to split each language on sensible boundaries (function/class edges)
# instead of just cutting every N characters mid-statement.
LC_LANGUAGE = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
}


def list_source_files(repo_root: str) -> list:
    found = []                                                            # repo-relative paths we'll return
    for current_dir, dirnames, filenames in os.walk(repo_root):           # standard recursive directory walk
        # mutating dirnames IN PLACE stops os.walk from ever descending into skip-dirs — much faster than
        # walking in and filtering the results afterward
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            full_path = os.path.join(current_dir, filename)               # absolute path on disk
            rel_path = os.path.relpath(full_path, repo_root)              # path relative to the repo root
            if detect_language(rel_path) is not None:                     # only files we know how to chunk
                found.append(rel_path)
    return found


def chunk_and_index(repo_root: str, persist_dir: str):
    all_docs = []                                                          # every chunk from every file, collected

    for rel_path in list_source_files(repo_root):                          # loop over every source file we found
        full_path = os.path.join(repo_root, rel_path)                      # rebuild the absolute path to read it
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:  # tolerate odd/binary-ish encodings
                content = f.read()
        except OSError:
            continue                                                        # unreadable file — skip, don't crash the run

        if not content.strip():                                             # empty file, nothing useful to chunk
            continue

        language = detect_language(rel_path)                                # e.g. "python"
        lc_language = LC_LANGUAGE.get(language)                             # the matching LangChain enum value

        # from_language() tries to break chunks on function/class boundaries
        # before falling back to plain newlines — this is the "chunking" half of RAG.
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lc_language, chunk_size=1200, chunk_overlap=100
        )
        chunks = splitter.split_text(content)                               # list[str], one entry per chunk

        for i, chunk_text in enumerate(chunks):                             # attach metadata to every chunk
            all_docs.append(Document(
                page_content=chunk_text,                                    # the actual code text
                metadata={
                    "file_path": rel_path,                                  # which file this came from
                    "language": language,                                   # useful if we ever want to filter
                    "chunk_index": i,                                       # position within the file
                },
            ))

    # Chroma.from_documents does two things at once: embeds every chunk via
    # the embeddings object we pass in, and writes the result to disk at
    # persist_directory — this is the "embedding + vector database" half.
    Chroma.from_documents(
        documents=all_docs,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )

    return len(all_docs), all_docs                                          # caller wants the count + the raw docs
