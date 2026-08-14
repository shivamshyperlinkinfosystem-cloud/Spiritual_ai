"""
Bhagavad Gita AI — Streamlit UI
Run:  streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Bhagavad Gita AI", page_icon="🕉", layout="centered")

BASE_DIR   = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# ── Load backend once ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_backend():
    from geeta_chat import build_app, FastEmbeddings, EMBED_MODEL
    from langchain_chroma import Chroma

    embeddings  = FastEmbeddings(EMBED_MODEL)
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    app         = build_app(vectorstore, embeddings)
    return app


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []   # list[BaseMessage]  — LangGraph history
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []   # list[dict]         — for rendering


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕉 Bhagavad Gita AI")
    st.markdown(
        "Ask anything about the **Bhagavad Gita**. "
        "Answers grounded in the original Sanskrit text."
    )
    st.divider()
    st.markdown("**LLM** `llama-3.3-70b-versatile` · Groq")
    st.markdown("**Search** Hybrid · vector + BM25")
    st.markdown("**Retrieval** Query rewriting + compression")
    st.markdown("**DB** ChromaDB · 919 passages")
    st.divider()

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_display = []
        st.rerun()

    st.divider()
    st.caption("Phase 1 · Spiritual AI · AI Virtual Guru\nLangGraph · Groq · ChromaDB")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center'>🕉 Bhagavad Gita — AI Virtual Guru</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:gray'>"
    "Ask anything · Answers grounded in the Sanskrit text"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Setup gates ───────────────────────────────────────────────────────────────
if not CHROMA_DIR.exists():
    st.error("**Knowledge base not found.**\n\nRun:\n```\npython ingest.py\n```")
    st.stop()

# Read from Streamlit secrets (cloud) or environment variable (local)
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key   # make it available to langchain_groq
else:
    st.error(
        "**GROQ_API_KEY not set.**\n\n"
        "**Local:** `export GROQ_API_KEY=gsk_...`\n\n"
        "**Streamlit Cloud:** add it under *Settings → Secrets*"
    )
    st.stop()

app = load_backend()

# ── Render chat history ───────────────────────────────────────────────────────
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"], avatar=entry["avatar"]):
        st.markdown(entry["content"])
        if entry.get("sources"):
            with st.expander("📖 Sources", expanded=False):
                st.caption(f"References: **{', '.join(entry['sources'])}**")

# ── Welcome ───────────────────────────────────────────────────────────────────
if not st.session_state.chat_display:
    with st.chat_message("assistant", avatar="🕉"):
        st.markdown(
            "Namaste 🙏 I am your guide to the wisdom of the **Bhagavad Gita**.\n\n"
            "You may ask me about:\n"
            "- **Specific shlokas** — *\"What is Chapter 2, Verse 47?\"*\n"
            "- **Core concepts** — Karma Yoga, Bhakti, Jnana, Dharma, Moksha\n"
            "- **Krishna's teachings** — on duty, action, and liberation\n"
            "- **Sanskrit meanings** — I will explain the original text\n\n"
            "*Ask your question below.*"
        )

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask the Bhagavad Gita…")

if user_input:
    # Show user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.chat_display.append({
        "role": "user", "avatar": "🧑", "content": user_input
    })

    # Add to LangGraph history
    st.session_state.messages.append(HumanMessage(content=user_input))

    initial_state = {
        "messages":    st.session_state.messages,
        "context":     "",
        "sources":     [],
        "is_relevant": True,
        "query":       "",
    }

    # ── Streaming response with live step indicators ──────────────────────────
    with st.chat_message("assistant", avatar="🕉"):
        step_ph  = st.empty()   # shows current pipeline step
        ans_ph   = st.empty()   # streams the answer
        full_content    = ""
        display_content = ""

        # What to show AFTER each node completes
        # (None = clear the step indicator)
        STEP_LABELS = {
            "rewrite":  "📖  Searching the Gita…",
            "retrieve": None,
        }

        step_ph.caption("🔍  Checking relevance…")

        for mode, data in app.stream(initial_state,
                                     stream_mode=["updates", "messages"]):
            # ── Node completed ────────────────────────────────────────────────
            if mode == "updates":
                node = next(iter(data))

                if node == "guard":
                    is_rel = data["guard"].get("is_relevant", True)
                    step_ph.caption(
                        "✍️  Understanding your question…" if is_rel
                        else "🤔  Checking Gita scope…"
                    )

                elif node in STEP_LABELS:
                    nxt = STEP_LABELS[node]
                    step_ph.caption(nxt) if nxt else step_ph.empty()

            # ── Token / message from an LLM node ─────────────────────────────
            elif mode == "messages":
                chunk, meta = data
                node    = meta.get("langgraph_node", "")
                content = getattr(chunk, "content", "")
                if not content:
                    continue

                if node == "generate":
                    full_content += content
                    display_content = full_content.split("\n\n*— Sources:")[0]
                    ans_ph.markdown(display_content + "▌")

                elif node == "reject":
                    step_ph.empty()
                    full_content    = content
                    display_content = content
                    ans_ph.markdown(display_content)

        # ── Final render (remove cursor) ──────────────────────────────────────
        step_ph.empty()
        ans_ph.markdown(display_content)

        # Extract sources from inline footnote
        sources = []
        if "\n\n*— Sources:" in full_content:
            raw     = full_content.split("\n\n*— Sources:")[1].rstrip("*").strip()
            sources = [s.strip() for s in raw.split(",") if s.strip()]

        if sources:
            with st.expander("📖 Sources", expanded=False):
                st.caption(f"References: **{', '.join(sources)}**")

    # Update LangGraph message history — append AI reply directly (no second invoke)
    from langchain_core.messages import AIMessage
    st.session_state.messages = list(st.session_state.messages) + [
        AIMessage(content=full_content)
    ]

    # Persist for re-render
    st.session_state.chat_display.append({
        "role":    "assistant",
        "avatar":  "🕉",
        "content": display_content,
        "sources": sources,
    })
