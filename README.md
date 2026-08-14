# 🕉 Spiritual AI — AI Virtual Guru

Phase 1 of a multi-persona spiritual guidance system powered by RAG (Retrieval-Augmented Generation). Answers questions grounded in the Bhagavad Gita, Yoga Sutras of Patanjali, and Ten Principal Upanishads.

## Features
- **6 Guru Personas** — General, Bhakti, Yoga, Meditation, Karma & Dharma, Healing
- **Hybrid search** — vector (ChromaDB MMR) + keyword (BM25) retrieval
- **Source grounding** — every answer cites chapter/verse/page
- **Streaming responses** with live step indicators
- **Topic guard** — rejects non-spiritual questions
- **Query rewriting** — handles multi-turn follow-ups

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build knowledge base (run once)
python scripts/ingest.py

# 3. Run the app
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit entry point
├── src/
│   ├── config.py             # paths + model config
│   ├── prompts.py            # all LLM prompts & personas  ← edit here
│   └── pipeline/
│       ├── state.py          # LangGraph state
│       ├── embeddings.py     # FastEmbeddings (ONNX)
│       ├── retriever.py      # BM25 + compression
│       ├── nodes.py          # 5 pipeline nodes
│       └── graph.py          # build_app()
├── data/
│   ├── pdfs/                 # source PDFs
│   ├── vectorstore/          # ChromaDB (pre-built)
│   └── chunks.json           # BM25 index (pre-built)
├── scripts/
│   └── ingest.py             # PDF → ChromaDB pipeline
└── docs/
    └── TECHNICAL_DOCUMENT.md
```

## Stack
`LangGraph` · `Groq (llama-3.3-70b)` · `ChromaDB` · `fastembed ONNX` · `Streamlit`
