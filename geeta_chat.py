"""
Bhagavad Gita AI — LangGraph + RAG + Groq

Improvements over v1:
  1. Query rewriting   — rewrites follow-up questions into standalone search queries
  2. Hybrid search     — vector (Chroma MMR) + keyword (BM25) combined
  3. Compression       — filters retrieved chunks by embedding similarity to query
  4. Chapter/verse     — cites Ch.X V.Y when metadata is available
  5. Grounding         — system prompt requires explicit sourcing of every claim
  6. Topic guard       — rejects unrelated questions before any retrieval

Graph:
    START → guard → (relevant)  → rewrite → retrieve → generate → END
                  → (unrelated) → reject  → END

Run:
    python geeta_chat.py
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

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CHROMA_DIR  = BASE_DIR / "chroma_db"
CHUNKS_PATH = BASE_DIR / "chunks.json"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"
RETRIEVAL_K = 8    # fetch more, then compress down
FINAL_K     = 5    # keep top-N after compression
SIM_THRESHOLD = 0.30  # minimum cosine similarity to keep a chunk

# ── Prompts ───────────────────────────────────────────────────────────────────
GUARD_SYSTEM = """\
You are a topic classifier for a Bhagavad Gita guidance system.
Classify the user's input as RELEVANT or IRRELEVANT.

RELEVANT — includes ALL of the following:
  • Direct Gita questions: shlokas, Sanskrit verses, chapters, Krishna, Arjuna,
    commentaries, Mahabharata, Vedic/Hindu philosophy.
  • Spiritual concepts: Karma Yoga, Bhakti Yoga, Jnana Yoga, Dharma, Atman,
    Brahman, Moksha, Maya, meditation, self-realisation.
  • Personal life struggles that the Gita speaks to — THESE ARE ALWAYS RELEVANT:
    confusion about purpose or goal, lack of focus or motivation, anxiety or fear,
    dealing with failure or disappointment, questions about duty and right action,
    how to act without attachment to results, overcoming grief or despair,
    finding meaning, inner peace, self-discipline, dealing with temptation,
    relationships and detachment, fear of death, identity questions.
  The Bhagavad Gita is a guide for human life — any genuine human struggle
  can be answered through its teachings.

IRRELEVANT — only clear off-topic questions:
  coding / programming, science / math, weather, cooking recipes, sports scores,
  news / current events, medical diagnosis, legal advice, financial tips,
  entertainment recommendations (movies, music), geography facts with no
  spiritual angle.

When in doubt, classify as RELEVANT.

Reply with exactly one word: RELEVANT or IRRELEVANT"""

REWRITE_PROMPT = """\
Given the conversation history and the latest question about the Bhagavad Gita, \
rewrite the question as a concise standalone search query that captures all context \
needed to retrieve the relevant shloka or passage.
- Include key Sanskrit/spiritual terms from earlier turns if referenced
- Keep it brief (1 sentence)
- Output only the search query, nothing else

Conversation history:
{history}

Latest question: {question}

Standalone search query:"""

REJECT_MESSAGE = """\
🙏 I am here solely to guide you through the wisdom of the **Bhagavad Gita**.

Your question appears to be outside its scope. Please ask me about:
- **Shlokas & chapters** — e.g. *"What does Chapter 2, Verse 47 say?"*
- **Core teachings** — Karma Yoga, Bhakti Yoga, Jnana Yoga
- **Key concepts** — Dharma, Atman, Brahman, Moksha, Maya
- **Sanskrit meanings** — I will explain the original text
- **Krishna's guidance** to Arjuna on duty, action, and liberation

*What would you like to learn from the Gita?*"""

SYSTEM_PROMPT = """\
You are a compassionate Acharya (spiritual teacher) and Sanskrit scholar of the \
Bhagavad Gita. You guide both scholars seeking textual knowledge AND seekers facing \
real-life struggles — because the Gita was spoken precisely for someone in crisis.

Rules:
1. When the user asks a personal life question (focus, purpose, anxiety, failure, \
   duty, attachment, grief), connect their struggle directly to the Gita's teaching. \
   Show them how Krishna's words apply to their situation. This is the Gita's highest use.
2. Base every answer on the retrieved passages below. Quote the shloka when present.
3. Always cite Adhyaya (chapter) and Shloka number when identifiable.
4. If the retrieved context does not cover a point, say: \
   "This is not in the retrieved passages, but the Gita teaches..." and give \
   the relevant wisdom briefly.
5. Be warm, practical, and encouraging — like a wise teacher, not a search engine.

--- Retrieved passages from the Bhagavad Gita ---
{context}
---"""


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
    """Keyword-based retriever using BM25Okapi over the saved chunks."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        tokenized = [d.page_content.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def invoke(self, query: str, k: int = RETRIEVAL_K) -> list[Document]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-k:][::-1]
        return [self.docs[i] for i in top_idx if scores[i] > 0]

    @classmethod
    def from_json(cls, path: Path) -> "BM25Retriever | None":
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in data]
        return cls(docs)


# ── Contextual compression ────────────────────────────────────────────────────
def compress(query: str, docs: list[Document], embeddings: FastEmbeddings) -> list[Document]:
    """Keep only docs whose cosine similarity to the query exceeds SIM_THRESHOLD."""
    if not docs:
        return docs
    q_emb  = np.array(embeddings.embed_query(query))
    d_embs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-9)
    sims   = d_norm @ q_norm
    ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked if s >= SIM_THRESHOLD][:FINAL_K]


# ── LangGraph state ───────────────────────────────────────────────────────────
class GeetaState(TypedDict):
    messages:    Annotated[Sequence[BaseMessage], add_messages]
    context:     str
    sources:     list[str]
    is_relevant: bool
    query:       str    # rewritten standalone query


# ── Guard node ────────────────────────────────────────────────────────────────
def make_guard_node(llm):
    def guard(state: GeetaState) -> dict:
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        if last_user is None:
            return {"is_relevant": True}
        resp = llm.invoke([
            SystemMessage(content=GUARD_SYSTEM),
            HumanMessage(content=last_user.content),
        ])
        return {"is_relevant": "IRRELEVANT" not in resp.content.strip().upper()}
    return guard


# ── Reject node ───────────────────────────────────────────────────────────────
def reject(state: GeetaState) -> dict:
    return {"messages": [AIMessage(content=REJECT_MESSAGE)]}


# ── Query rewrite node ────────────────────────────────────────────────────────
def make_rewrite_node(llm):
    def rewrite(state: GeetaState) -> dict:
        msgs = list(state["messages"])
        last_user = next(
            (m for m in reversed(msgs) if isinstance(m, HumanMessage)), None
        )
        if last_user is None:
            return {"query": ""}

        prev = [m for m in msgs if m is not last_user]
        if not prev:                        # first turn — no rewriting needed
            return {"query": last_user.content}

        history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
            for m in prev[-6:]              # last 3 turns
        )
        resp = llm.invoke(
            REWRITE_PROMPT.format(history=history, question=last_user.content)
        )
        query = resp.content.strip()
        return {"query": query or last_user.content}
    return rewrite


# ── Hybrid retrieve + compress node ──────────────────────────────────────────
def make_retrieve_node(vector_retriever, bm25: BM25Retriever | None, embeddings):
    def retrieve(state: GeetaState) -> dict:
        query = state.get("query") or next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if not query:
            return {"context": "", "sources": []}

        # 1. Vector search (MMR for diversity)
        vec_docs = vector_retriever.invoke(query)

        # 2. BM25 keyword search
        kw_docs = bm25.invoke(query) if bm25 else []

        # 3. Merge & deduplicate (vector docs first = higher priority)
        seen, combined = set(), []
        for doc in vec_docs + kw_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        # 4. Contextual compression
        filtered = compress(query, combined, embeddings)

        # 5. Build context + source citations
        parts, sources = [], []
        for i, doc in enumerate(filtered, 1):
            parts.append(f"[Passage {i}]\n{doc.page_content.strip()}")
            meta = doc.metadata
            chap, verse, page = meta.get("chapter"), meta.get("verse"), meta.get("page")
            if chap and verse:
                sources.append(f"Ch.{chap} V.{verse}")
            elif page is not None:
                sources.append(f"p.{int(page) + 1}")

        return {"context": "\n\n".join(parts), "sources": sorted(set(sources))}
    return retrieve


# ── Generate node ─────────────────────────────────────────────────────────────
def make_generate_node(llm, prompt_template):
    def generate(state: GeetaState) -> dict:
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


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_guard(state: GeetaState) -> str:
    return "rewrite" if state.get("is_relevant", True) else "reject"


# ── Build compiled LangGraph app ──────────────────────────────────────────────
def build_app(vectorstore, embeddings: FastEmbeddings):
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=2048)

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": 20},
    )
    bm25 = BM25Retriever.from_json(CHUNKS_PATH)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    g = StateGraph(GeetaState)
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


# ── Terminal chat loop ────────────────────────────────────────────────────────
def chat() -> None:
    console = Console()

    if not CHROMA_DIR.exists():
        console.print(Panel("[red]Run 'python ingest.py' first.[/red]", border_style="red"))
        sys.exit(1)

    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]Set GROQ_API_KEY before running.[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold yellow]🕉  Bhagavad Gita AI[/bold yellow]\n\n"
        "[dim]Hybrid search · Query rewriting · Contextual compression\n"
        "Type 'exit' · 'clear' to reset[/dim]",
        border_style="yellow",
    ))

    with console.status("[dim]Loading…[/dim]"):
        embeddings  = FastEmbeddings(EMBED_MODEL)
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        app         = build_app(vectorstore, embeddings)

    console.print("[dim]Ready.[/dim]\n")
    messages: list[BaseMessage] = []

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]🙏 Namaste![/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
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
        messages = list(result["messages"])

        console.print()
        console.print(Text("🕉  Gita AI:", style="bold green"))
        console.print(Markdown(messages[-1].content))
        console.print()


if __name__ == "__main__":
    chat()
