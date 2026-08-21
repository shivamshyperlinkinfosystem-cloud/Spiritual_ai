"""
graph.py — Compiles the LangGraph state machine.

Usage:
    from src.pipeline.graph import build_app
    app = build_app(vectorstore, embeddings, system_prompt)
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

from src.config import LLM_MODEL, RETRIEVAL_K, MMR_FETCH_K, CHUNKS_PATH
from src.pipeline.state import SpiritualState
from src.pipeline.embeddings import FastEmbeddings
from src.pipeline.retriever import BM25Retriever
from src.pipeline.nodes import (
    make_guard_node,
    make_chat_node,
    make_rewrite_node,
    make_retrieve_node,
    make_generate_node,
    reject,
    route_after_guard,
)
from src.prompts import PERSONAS, DEFAULT_PERSONA


def build_app(vectorstore, embeddings: FastEmbeddings,
              system_prompt: str | None = None):
    """
    Compile the LangGraph RAG pipeline for a given persona system prompt.

    Graph:
        START → guard → (relevant)  → rewrite → retrieve → generate → END
                      → (unrelated) → reject  → END
    """
    if system_prompt is None:
        system_prompt = PERSONAS[DEFAULT_PERSONA]["system"]

    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=2048)

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": MMR_FETCH_K},
    )
    bm25 = BM25Retriever.from_json(CHUNKS_PATH)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    g = StateGraph(SpiritualState)
    g.add_node("guard",    make_guard_node(llm))
    g.add_node("chat",     make_chat_node(llm, system_prompt))
    g.add_node("rewrite",  make_rewrite_node(llm))
    g.add_node("retrieve", make_retrieve_node(vector_retriever, bm25, embeddings))
    g.add_node("generate", make_generate_node(llm, prompt_template))
    g.add_node("reject",   reject)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"rewrite": "rewrite", "chat": "chat", "reject": "reject"})
    g.add_edge("chat",     END)
    g.add_edge("rewrite",  "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate",  END)
    g.add_edge("reject",    END)

    return g.compile()
