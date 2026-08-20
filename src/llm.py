"""
llm.py
------
Model wrappers, kept in one place so:
  1. every node imports from here instead of touching an SDK directly
  2. tests can monkeypatch these functions and skip real network/API calls

Both halves now run fully locally — no API key, no network call, no rate
limit, no cloud dependency at all once the models are pulled/downloaded once:
  - embeddings: HuggingFace/sentence-transformers (see get_embeddings)
  - chat/generation: Ollama, serving a local model (see get_chat_model)
"""

from langchain_huggingface import HuggingFaceEmbeddings   # local embeddings, no API involved
from langchain_ollama import ChatOllama                    # local chat model, served by Ollama


def get_embeddings() -> HuggingFaceEmbeddings:
    # BGE-small: ~130MB, runs on CPU, no GPU required. Downloaded once on first
    # use and cached locally after that (~/.cache/huggingface by default) — every
    # run after the first is fully offline, so there's no rate limit to hit here.
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_chat_model(model_name: str = "qwen2.5-coder:3b") -> ChatOllama:
    # qwen2.5-coder:3b: ~1.9GB, code-specialized, comfortably fits a small
    # local footprint. Requires Ollama running locally (default localhost:11434)
    # with the model already pulled: `ollama pull qwen2.5-coder:3b`.
    # temperature=0 because every call site here wants a consistent,
    # literal description of real code — not creative variation.
    return ChatOllama(model=model_name, temperature=0)


def chat_complete(prompt: str, model_name: str = "qwen2.5-coder:3b") -> str:
    llm = get_chat_model(model_name)    # build a fresh client scoped to this one call
    response = llm.invoke(prompt)       # send the prompt, get an AIMessage back
    return response.content             # callers only ever want the text