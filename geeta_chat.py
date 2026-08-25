"""
Spiritual AI — LangGraph RAG with 5 Guru personas.

Graph:
    START → guard → RELEVANT → rewrite → retrieve → generate → END
                  → UNRELATED → reject  → END
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from typing_extensions import TypedDict

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

# ── All prompts live in prompts.py — edit there, not here ─────────────────────
from prompts import (
    PERSONAS, DEFAULT_PERSONA,
    GUARD_SYSTEM, REWRITE_PROMPT, REJECT_MESSAGE,
    WELCOME_MESSAGES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CHROMA_DIR    = BASE_DIR / "chroma_db"
CHUNKS_PATH   = BASE_DIR / "chunks.json"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
RETRIEVAL_K   = 8
FINAL_K       = 5
SIM_THRESHOLD = 0.28


# ── (prompts removed — see prompts.py) ───────────────────────────────────────




# ── fastembed wrapper ─────────────────────────────────────────────────────────
class FastEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


# ── BM25 retriever ────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])

    def invoke(self, query: str, k: int = RETRIEVAL_K) -> list[Document]:
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[-k:][::-1]
        return [self.docs[i] for i in top_idx if scores[i] > 0]

    @classmethod
    def from_json(cls, path: Path):
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in data]
        return cls(docs)


# ── Contextual compression ────────────────────────────────────────────────────
def compress(query: str, docs: list[Document], embeddings: FastEmbeddings) -> list[Document]:
    if not docs:
        return docs
    q   = np.array(embeddings.embed_query(query))
    d   = np.array(embeddings.embed_documents([x.page_content for x in docs]))
    qn  = q / (np.linalg.norm(q) + 1e-9)
    dn  = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    sim = dn @ qn
    ranked = sorted(zip(docs, sim), key=lambda x: x[1], reverse=True)
    return [doc for doc, s in ranked if s >= SIM_THRESHOLD][:FINAL_K]


# ── LangGraph state ───────────────────────────────────────────────────────────
class SpiritualState(TypedDict):
    messages:    Annotated[Sequence[BaseMessage], add_messages]
    context:     str
    sources:     list[str]
    is_relevant: bool
    query:       str


# ── Nodes ─────────────────────────────────────────────────────────────────────
def make_guard_node(llm):
    def guard(state: SpiritualState) -> dict:
        last = next((m for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), None)
        if last is None:
            return {"is_relevant": True}
        resp = llm.invoke([SystemMessage(content=GUARD_SYSTEM),
                           HumanMessage(content=last.content)])
        return {"is_relevant": "IRRELEVANT" not in resp.content.strip().upper()}
    return guard


def reject(state: SpiritualState) -> dict:
    return {"messages": [AIMessage(content=REJECT_MESSAGE)]}


def make_rewrite_node(llm):
    def rewrite(state: SpiritualState) -> dict:
        msgs = list(state["messages"])
        last = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if last is None:
            return {"query": ""}
        prev = [m for m in msgs if m is not last]
        if not prev:
            return {"query": last.content}
        history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
            for m in prev[-6:]
        )
        resp = llm.invoke(
            REWRITE_PROMPT.format(history=history, question=last.content)
        )
        return {"query": resp.content.strip() or last.content}
    return rewrite


def make_retrieve_node(vector_retriever, bm25, embeddings):
    def retrieve(state: SpiritualState) -> dict:
        query = state.get("query") or next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)), ""
        )
        if not query:
            return {"context": "", "sources": []}

        vec_docs = vector_retriever.invoke(query)
        kw_docs  = bm25.invoke(query) if bm25 else []

        seen, combined = set(), []
        for doc in vec_docs + kw_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        filtered = compress(query, combined, embeddings)

        parts, sources = [], []
        for i, doc in enumerate(filtered, 1):
            parts.append(f"[Passage {i}]\n{doc.page_content.strip()}")
            m    = doc.metadata
            src  = m.get("source_text", m.get("source", "Scripture"))
            chap = m.get("chapter")
            vers = m.get("verse")
            page = m.get("page")
            if chap and vers:
                sources.append(f"{src} Ch.{chap} V.{vers}")
            elif page is not None:
                sources.append(f"{src} p.{int(page)+1}")
            else:
                sources.append(src)

        return {"context": "\n\n".join(parts), "sources": sorted(set(sources))}
    return retrieve


def make_generate_node(llm, prompt_template):
    def generate(state: SpiritualState) -> dict:
        formatted = prompt_template.invoke({
            "context":  state.get("context") or "(no passages retrieved)",
            "messages": state["messages"],
        })
        response = llm.invoke(formatted)
        sources  = state.get("sources", [])
        if sources:
            response.content += f"\n\n*— Sources: {', '.join(sources)}*"
        return {"messages": [response]}
    return generate


def route_after_guard(state: SpiritualState) -> str:
    return "rewrite" if state.get("is_relevant", True) else "reject"


# ── Build app (one per persona) ───────────────────────────────────────────────
def build_app(vectorstore, embeddings: FastEmbeddings,
              system_prompt: str | None = None):
    if system_prompt is None:
        system_prompt = PERSONAS[DEFAULT_PERSONA]["system"]

    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=2048)

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": 20},
    )
    bm25 = BM25Retriever.from_json(CHUNKS_PATH)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    g = StateGraph(SpiritualState)
    g.add_node("guard",    make_guard_node(llm))
    g.add_node("rewrite",  make_rewrite_node(llm))
    g.add_node("retrieve", make_retrieve_node(vector_retriever, bm25, embeddings))
    g.add_node("generate", make_generate_node(llm, prompt_template))
    g.add_node("reject",   reject)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"rewrite": "rewrite", "reject": "reject"})
    g.add_edge("rewrite",  "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate",  END)
    g.add_edge("reject",    END)

    return g.compile()


# ── Terminal chat ─────────────────────────────────────────────────────────────
def chat() -> None:
    console = Console()

    if not CHROMA_DIR.exists():
        console.print(Panel("[red]Run 'python ingest.py' first.[/red]", border_style="red"))
        sys.exit(1)
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]Set GROQ_API_KEY first.[/red]")
        sys.exit(1)

    # Pick persona
    persona_keys = list(PERSONAS.keys())
    console.print(Panel.fit(
        "[bold yellow]🕉  Spiritual AI — Choose your Guru[/bold yellow]\n\n"
        + "\n".join(f"  [{i+1}] {k}  —  {PERSONAS[k]['tagline']}"
                   for i, k in enumerate(persona_keys)),
        border_style="yellow",
    ))
    try:
        choice = int(console.input("\nEnter number (default 1): ").strip() or "1")
        persona_key = persona_keys[choice - 1]
    except (ValueError, IndexError):
        persona_key = DEFAULT_PERSONA

    console.print(f"\n[green]Guru selected: {persona_key}[/green]\n")

    with console.status("[dim]Loading knowledge base…[/dim]"):
        embeddings  = FastEmbeddings(EMBED_MODEL)
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR),
                             embedding_function=embeddings)
        app         = build_app(vectorstore, embeddings,
                                PERSONAS[persona_key]["system"])

    messages: list[BaseMessage] = []

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]🙏 Namaste![/yellow]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("\n[yellow]🙏 Namaste![/yellow]")
            break
        if user_input.lower() == "clear":
            messages = []
            console.print("[dim]Cleared.[/dim]\n")
            continue

        messages.append(HumanMessage(content=user_input))
        with console.status("[dim]Processing…[/dim]"):
            result   = app.invoke({
                "messages": messages, "context": "", "sources": [],
                "is_relevant": True, "query": "",
            })

        if result.get("context"):
            console.print(f"[dim]📖 Retrieved passages:\n{result['context'][:500]}…[/dim]\n")
        else:
            console.print("[yellow]⚠ No scripture retrieved — LLM answered from general knowledge[/yellow]\n")

        messages = list(result["messages"])
        console.print()
        console.print(Text(f"{persona_key.split()[0]}  Guru:", style="bold green"))
        console.print(Markdown(messages[-1].content))
        console.print()


if __name__ == "__main__":
    chat()
