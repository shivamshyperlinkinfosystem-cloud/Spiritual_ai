"""
nodes.py — All 5 LangGraph node factory functions.

Each factory returns a callable node that reads/writes SpiritualState.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.pipeline.state import SpiritualState
from src.pipeline.retriever import BM25Retriever, compress
from src.config import RETRIEVAL_K
from src.prompts import GUARD_SYSTEM, REWRITE_PROMPT, REJECT_MESSAGE, CHAT_SYSTEM_SUFFIX


# ── 1. Guard — topic relevance classifier ────────────────────────────────────
def make_guard_node(llm):
    def guard(state: SpiritualState) -> dict:
        last = next((m for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), None)
        if last is None:
            return {"is_relevant": True, "intent": "spiritual"}
        resp = llm.invoke([
            SystemMessage(content=GUARD_SYSTEM),
            HumanMessage(content=last.content),
        ])
        word = resp.content.strip().upper()
        if "GREETING" in word:
            intent = "greeting"
        elif "IRRELEVANT" in word:
            intent = "irrelevant"
        else:
            intent = "spiritual"
        return {"is_relevant": intent != "irrelevant", "intent": intent}
    return guard


# ── 1b. Chat — warm conversational reply (no RAG) ────────────────────────────
def make_chat_node(llm, system_prompt: str):
    chat_system = system_prompt.split("\n\nRules:")[0] + CHAT_SYSTEM_SUFFIX

    def chat(state: SpiritualState) -> dict:
        recent = list(state["messages"])[-10:]
        response = llm.invoke([SystemMessage(content=chat_system), *recent])
        return {"messages": [response], "sources": []}
    return chat


# ── 2. Reject — polite off-topic refusal ─────────────────────────────────────
def reject(state: SpiritualState) -> dict:
    return {"messages": [AIMessage(content=REJECT_MESSAGE)]}


# ── 3. Rewrite — standalone search query builder ─────────────────────────────
def make_rewrite_node(llm):
    def rewrite(state: SpiritualState) -> dict:
        msgs = list(state["messages"])
        last = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if last is None:
            return {"query": ""}
        prev = [m for m in msgs if m is not last]
        if not prev:                                   # first turn — no rewriting needed
            return {"query": last.content}
        history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
            for m in prev[-6:]                         # last 3 turns max
        )
        resp = llm.invoke(
            REWRITE_PROMPT.format(history=history, question=last.content)
        )
        return {"query": resp.content.strip() or last.content}
    return rewrite


# ── 4. Retrieve — hybrid vector + BM25 with compression ──────────────────────
def make_retrieve_node(vector_retriever, bm25: BM25Retriever | None, embeddings):
    def retrieve(state: SpiritualState) -> dict:
        query = state.get("query") or next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)), ""
        )
        if not query:
            return {"context": "", "sources": []}

        vec_docs = vector_retriever.invoke(query)
        kw_docs  = bm25.invoke(query) if bm25 else []

        # Merge + deduplicate (vector docs have priority)
        seen, combined = set(), []
        for doc in vec_docs + kw_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        filtered = compress(query, combined, embeddings)

        # Build context string + source citations
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
                sources.append(f"{src} p.{int(page) + 1}")
            else:
                sources.append(src)

        return {
            "context": "\n\n".join(parts),
            "sources": sorted(set(sources)),
        }
    return retrieve


# ── 5. Generate — LLM answer with persona ────────────────────────────────────
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


# ── Routing helper ────────────────────────────────────────────────────────────
def route_after_guard(state: SpiritualState) -> str:
    intent = state.get("intent", "spiritual")
    if intent == "irrelevant":
        return "reject"
    if intent == "greeting":
        return "chat"
    return "rewrite"
