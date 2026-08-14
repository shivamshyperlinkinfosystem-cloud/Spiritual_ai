"""
Spiritual AI — Streamlit UI with 5 Guru personas.
Run:  streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Spiritual AI", page_icon="🕉", layout="centered")

BASE_DIR   = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_base():
    """Embeddings + vectorstore — loaded once, shared across all personas."""
    from geeta_chat import FastEmbeddings, EMBED_MODEL
    from langchain_chroma import Chroma
    emb = FastEmbeddings(EMBED_MODEL)
    vs  = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
    return emb, vs


@st.cache_resource(show_spinner="Preparing Guru…")
def load_persona_app(persona_key: str):
    """Compile LangGraph app for one persona (cached per persona key)."""
    from geeta_chat import build_app, PERSONAS
    emb, vs = load_base()
    return build_app(vs, emb, PERSONAS[persona_key]["system"])


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "persona"      not in st.session_state:
    st.session_state.persona      = "🕉  General Guru"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕉 Spiritual AI")
    st.markdown("*AI Virtual Guru — Phase 1*")
    st.divider()

    from prompts import PERSONAS, WELCOME_MESSAGES
    persona_keys = list(PERSONAS.keys())

    st.markdown("### Choose Your Guru")
    selected = st.radio(
        label="Guru",
        options=persona_keys,
        index=persona_keys.index(st.session_state.persona),
        label_visibility="collapsed",
        format_func=lambda k: f"{k}\n_{PERSONAS[k]['tagline']}_",
    )

    # Reset conversation when persona changes
    if selected != st.session_state.persona:
        st.session_state.persona      = selected
        st.session_state.messages     = []
        st.session_state.chat_display = []
        st.rerun()

    st.divider()
    st.markdown(f"**Knowledge base**")
    st.caption("• Bhagavad Gita")
    st.caption("• Yoga Sutras of Patanjali")
    st.caption("• Ten Principal Upanishads")
    st.divider()
    st.markdown("**LLM** `llama-3.3-70b-versatile`")
    st.markdown("**Search** Hybrid · vector + BM25")

    st.divider()
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_display = []
        st.rerun()

    st.divider()
    st.caption("Phase 1 · Spiritual AI · LangGraph + Groq")


# ── Header ────────────────────────────────────────────────────────────────────
persona_info = PERSONAS[st.session_state.persona]
icon = st.session_state.persona.split()[0]

st.markdown(
    f"<h2 style='text-align:center'>{st.session_state.persona}</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:gray'>{persona_info['tagline']}</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Setup gates ───────────────────────────────────────────────────────────────
if not CHROMA_DIR.exists():
    st.error("**Knowledge base not found.**\n\nRun:\n```\npython ingest.py\n```")
    st.stop()

groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
else:
    st.error("**GROQ_API_KEY not set.**\n\nLocal: `export GROQ_API_KEY=gsk_...`")
    st.stop()

app = load_persona_app(st.session_state.persona)

# ── Render chat history ───────────────────────────────────────────────────────
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"], avatar=entry["avatar"]):
        st.markdown(entry["content"])
        if entry.get("sources"):
            with st.expander("📖 Sources", expanded=False):
                for src in entry["sources"]:
                    st.caption(f"• {src}")

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.chat_display:
    with st.chat_message("assistant", avatar=icon):
        st.markdown(WELCOME_MESSAGES.get(st.session_state.persona,
                                         "Namaste 🙏 How can I guide you today?"))

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(f"Ask {st.session_state.persona.strip()}…")

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.chat_display.append({
        "role": "user", "avatar": "🧑", "content": user_input
    })
    st.session_state.messages.append(HumanMessage(content=user_input))

    initial_state = {
        "messages":    st.session_state.messages,
        "context":     "",
        "sources":     [],
        "is_relevant": True,
        "query":       "",
    }

    # ── Streaming with step indicators ───────────────────────────────────────
    with st.chat_message("assistant", avatar=icon):
        step_ph  = st.empty()
        ans_ph   = st.empty()
        full_content    = ""
        display_content = ""

        STEP_LABELS = {
            "rewrite":  "📖  Searching sacred texts…",
            "retrieve": None,
        }
        step_ph.caption("🔍  Checking relevance…")

        for mode, data in app.stream(initial_state,
                                     stream_mode=["updates", "messages"]):
            if mode == "updates":
                node = next(iter(data))
                if node == "guard":
                    is_rel = data["guard"].get("is_relevant", True)
                    step_ph.caption(
                        "✍️  Understanding your question…" if is_rel
                        else "🤔  Checking scope…"
                    )
                elif node in STEP_LABELS:
                    nxt = STEP_LABELS[node]
                    if nxt:
                        step_ph.caption(nxt)
                    else:
                        step_ph.empty()

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

        step_ph.empty()
        ans_ph.markdown(display_content)

        # Sources
        sources = []
        if "\n\n*— Sources:" in full_content:
            raw     = full_content.split("\n\n*— Sources:")[1].rstrip("*").strip()
            sources = [s.strip() for s in raw.split(",") if s.strip()]

        if sources:
            with st.expander("📖 Sources", expanded=False):
                for src in sources:
                    st.caption(f"• {src}")

    # Update session state
    st.session_state.messages = list(st.session_state.messages) + [
        AIMessage(content=full_content)
    ]
    st.session_state.chat_display.append({
        "role":    "assistant",
        "avatar":  icon,
        "content": display_content,
        "sources": sources,
    })
