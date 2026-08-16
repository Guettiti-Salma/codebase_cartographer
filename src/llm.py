"""
llm.py
------
Thin wrappers around every Gemini API call, kept in one place so:
  1. every node imports from here instead of touching the SDK directly
  2. tests can monkeypatch these two functions and skip real network/API calls
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Gemini via LangChain


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Google's current text embedding model — on the free tier with a very
    # high tokens-per-minute limit, which is what makes indexing a whole
    # repo practical without paying anything.
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_chat_model(model_name: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    # temperature=0 because every call site here wants a consistent,
    # literal description of real code — not creative variation.
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


def chat_complete(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    llm = get_chat_model(model_name)    # build a fresh client scoped to this one call
    response = llm.invoke(prompt)       # send the prompt, get an AIMessage back
    return response.content             # callers only ever want the text
