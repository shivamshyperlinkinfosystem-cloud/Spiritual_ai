# Spiritual AI — Technical Document
**Phase 1: AI Virtual Guru**
Version 1.0 | Project: Spiritual AI | Stack: LangGraph · Groq · ChromaDB · Streamlit

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [LangGraph Pipeline (Deep Dive)](#4-langgraph-pipeline-deep-dive)
5. [RAG — How It Works](#5-rag--how-it-works)
6. [Knowledge Base](#6-knowledge-base)
7. [5 Guru Personas](#7-5-guru-personas)
8. [File Structure](#8-file-structure)
9. [Data Flow — End to End](#9-data-flow--end-to-end)
10. [API & Model Details](#10-api--model-details)
11. [Streamlit UI](#11-streamlit-ui)
12. [Setup & Run](#12-setup--run)
13. [Deployment on Streamlit Cloud](#13-deployment-on-streamlit-cloud)
14. [Phase 1 Completion Status](#14-phase-1-completion-status)
15. [What to Build Next (Phase 2)](#15-what-to-build-next-phase-2)

---

## 1. Project Overview

**Spiritual AI** is an AI-powered virtual spiritual guide that answers questions about
sacred Hindu texts — the Bhagavad Gita, Yoga Sutras of Patanjali, and Ten Principal
Upanishads — using Retrieval-Augmented Generation (RAG).

The system supports **5 specialized Guru personas**, each with a different focus,
tone, and teaching style. Every answer is grounded in actual passages from the
source texts with precise citations.

### What a user can do
- Ask questions about shlokas, sutras, and Upanishadic teachings
- Share personal struggles (focus, anxiety, grief, duty) and receive guidance
  through scriptural wisdom
- Switch between 5 different Guru personas
- See which text and which page/verse each answer comes from

### What the system does NOT do
- Answer unrelated questions (coding, weather, sports, etc.)
- Fabricate answers — if the text doesn't cover it, it says so explicitly

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | `llama-3.3-70b-versatile` via Groq API | Generates spiritual guidance answers |
| **Orchestration** | LangGraph 1.2 | State machine managing the 5-node pipeline |
| **Vector DB** | ChromaDB | Stores and searches text embeddings |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` via fastembed (ONNX) | Converts text to vectors — handles Sanskrit/Devanagari |
| **Keyword Search** | BM25 (rank_bm25) | Catches exact Sanskrit terms embeddings miss |
| **PDF Loading** | pypdf | Extracts text from sacred text PDFs |
| **UI** | Streamlit 1.61 | Web chat interface |
| **LangChain** | langchain-core, langchain-chroma, langchain-groq | RAG plumbing and prompt management |

### Why Groq?
Groq's LPU (Language Processing Unit) hardware runs LLaMA at extremely high speed
(~750 tokens/second). This is critical for achieving near sub-2-second response times.

### Why fastembed (ONNX) instead of sentence-transformers (PyTorch)?
The `sentence-transformers` library requires `torchvision` which was not available
on the deployment machine. `fastembed` runs the same model through ONNX Runtime —
no GPU, no torchvision — and is lighter to deploy.

### Why BM25 + Vector (Hybrid Search)?
- **Vector search** is good for semantic similarity ("what does the Gita say about peace?")
- **BM25 keyword search** is good for exact terms ("Nishkama Karma", "Pratyahara", "Samadhi")
- Combining both gives better coverage than either alone

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                            │
│                    Streamlit Web App                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ User question
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                        │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐   │
│  │  guard  │───▶│ rewrite │───▶│ retrieve │───▶│ generate │   │
│  └─────────┘    └─────────┘    └──────────┘    └──────────┘   │
│       │                                                          │
│       └──(off-topic)──▶ reject                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────┐       ┌──────────────────┐
   │   ChromaDB      │       │   Groq API        │
   │  Vector Store   │       │  llama-3.3-70b    │
   │  2046 chunks    │       │  (LLM answering)  │
   └─────────────────┘       └──────────────────┘
              ▲
   ┌──────────┴──────────┐
   │   BM25 Index        │
   │   chunks.json       │
   └─────────────────────┘
```

---

## 4. LangGraph Pipeline (Deep Dive)

LangGraph is a framework for building stateful AI applications as a directed graph.
Each node is a function that reads from and writes to a shared **State** object.

### State Object

```python
class SpiritualState(TypedDict):
    messages:    list[BaseMessage]  # full conversation history
    context:     str                # retrieved passages for current question
    sources:     list[str]          # citation strings (e.g. "Bhagavad Gita Ch.2 V.47")
    is_relevant: bool               # set by guard — True = proceed, False = reject
    query:       str                # rewritten standalone search query
```

### Graph Flow

```
START
  │
  ▼
[Node 1: guard]
  Asks the LLM: "Is this about spiritual wisdom? RELEVANT or IRRELEVANT"
  Sets state.is_relevant = True/False
  │
  ├── RELEVANT ──────────────────────────────────────────────────┐
  │                                                               │
  ▼                                                               │
[Node 2: rewrite]                                                 │
  Rewrites follow-up questions into standalone search queries.    │
  Example: "what does it mean?" → "meaning of Nishkama Karma     │
  as taught in Bhagavad Gita Chapter 3"                          │
  Sets state.query                                                │
  │                                                               │
  ▼                                                               │
[Node 3: retrieve]                                                │
  Step A: Vector search (ChromaDB MMR) — top 8 semantic matches  │
  Step B: BM25 keyword search — top 8 keyword matches            │
  Step C: Merge + deduplicate                                     │
  Step D: Compress — keep only chunks with cosine similarity ≥ 0.28
  Result: 5 most relevant passages with source citations          │
  Sets state.context and state.sources                            │
  │                                                               │
  ▼                                                               │
[Node 4: generate]                                                │
  Sends system prompt (persona) + retrieved context               │
  + conversation history → Groq LLM                              │
  Appends source citations to response                            │
  Sets state.messages (appends AI reply)                          │
  │                                                               │
  └───────────────────────────────────────────────────────────────┤
  │                                                               │
  └── IRRELEVANT ─────────────────────────────────────────────── │
  │                                                               │
  ▼                                                               │
[Node 5: reject]                                                  │
  Returns polite refusal message                                  │
  Sets state.messages (appends rejection AIMessage)               │
  │                                                               │
  ▼                                                               │
END ◄─────────────────────────────────────────────────────────────┘
```

### Why LangGraph instead of a simple function?

| Simple function | LangGraph |
|---|---|
| Hard to add/remove steps | Add/remove nodes without touching other code |
| No conditional routing | Conditional edges (relevant/irrelevant path) |
| No streaming support | `stream_mode=["updates","messages"]` built-in |
| State is manual | State managed automatically with reducers |
| Hard to debug | Each node's input/output is traceable |

---

## 5. RAG — How It Works

RAG = **Retrieval Augmented Generation**. Instead of asking the LLM from memory,
we first *retrieve* the relevant text passages and *augment* the LLM's prompt with
them. This grounds answers in the actual texts.

### Step-by-step

```
Question: "What does the Gita say about doing work without attachment?"

STEP 1 — EMBED THE QUESTION
  "doing work without attachment" 
  → [0.23, -0.14, 0.87, ...] (384-dimensional vector)

STEP 2 — SEARCH ChromaDB (Vector Search)
  Find the 8 chunks whose vectors are most similar to the question vector
  Result: passages from Bhagavad Gita Ch.3, Ch.5, Ch.18 about karma yoga

STEP 3 — BM25 KEYWORD SEARCH  
  Find chunks containing exact words: "work", "attachment", "nishkama"
  May catch different passages than vector search

STEP 4 — MERGE + COMPRESS
  Combine vector + BM25 results (deduplicate)
  Filter: keep only chunks with cosine similarity ≥ 0.28
  Keep top 5

STEP 5 — GENERATE
  System prompt (persona) + retrieved passages + user question
  → Groq LLM → "Chapter 3, verse 19: 'Tasmad asaktah satatam...'"

STEP 6 — CITE
  Append sources: "Bhagavad Gita Ch.3 p.45"
```

### Contextual Compression
After retrieval, each chunk is scored by cosine similarity to the query.
Chunks below the threshold (0.28) are removed. This means only the
truly relevant passages reach the LLM — reducing noise and improving
answer quality.

### Hybrid Search Scoring

```
Final results = deduplicate(
    vector_search(query, k=8)   ← semantic understanding
    + bm25_search(query, k=8)   ← exact keyword matching
)
then compress to top 5 by similarity
```

---

## 6. Knowledge Base

### Source Texts

| Text | Pages | Chunks | Focus |
|---|---|---|---|
| **Bhagavad Gita** | 170 | 919 | Karma, Dharma, Bhakti, Jnana, all yoga |
| **Yoga Sutras of Patanjali** | 239 | 778 | 8-limbed yoga, samadhi, meditation |
| **Ten Principal Upanishads** | 149 | 349 | Atman, Brahman, consciousness, non-duality |
| **Total** | **558** | **2046** | |

### How PDFs become searchable chunks

```
PDF File
  │
  ▼  (pypdf)
Raw text per page  →  metadata added: {page, source_text, chapter, verse}
  │
  ▼  (RecursiveCharacterTextSplitter)
Chunks of ~800 characters with 150-char overlap
Sanskrit-aware separators: ॥  ।  \n\n  \n  (space)
  │
  ├──▶ chunks.json     (saved for BM25 keyword search)
  │
  ▼  (fastembed ONNX)
384-dimensional embedding vectors
  │
  ▼  (ChromaDB)
Stored in chroma_db/ with metadata
```

### Adding new PDFs
Drop any `.pdf` file into the project folder and run:
```bash
python ingest.py
```
The system auto-discovers all PDFs, assigns source names, and rebuilds the
entire knowledge base.

### Source name detection
The ingest script automatically names sources from filenames:

| Filename | Detected Source |
|---|---|
| `final_geeta.pdf` | Bhagavad Gita |
| `Patanjali-yogasutra.IGS.pdf` | Yoga Sutras of Patanjali |
| `The-Ten-Principal-Upanishads.pdf` | Ten Principal Upanishads |
| any other `.pdf` | Prettified filename |

---

## 7. Five Guru Personas

Each persona is a different **system prompt** injected at the top of every
LLM call. The RAG pipeline and knowledge base remain the same for all personas.

| Persona | Focus | Tone | Primary Texts Used |
|---|---|---|---|
| 🕉 **General Guru** | Balanced across all texts | Warm, accessible | All three |
| 🙏 **Bhakti Guru** | Devotion, love, surrender | Heart-centred, devotional | Gita Ch.9,12,18 |
| 🧘 **Yoga Guru** | 8 limbs, practice, discipline | Systematic, practical | Yoga Sutras |
| 🔮 **Meditation Guru** | Consciousness, Atman, silence | Contemplative, still | Upanishads |
| ⚖️ **Karma & Dharma Guru** | Right action, duty, ethics | Direct, grounded | Gita Ch.2,3,5 |
| 💚 **Healing Guru** | Grief, anxiety, emotional pain | Compassionate, gentle | Gita Ch.2,18 + Upanishads |

### How persona switching works
1. User selects a persona in the Streamlit sidebar
2. Streamlit detects the change and clears conversation history
3. `load_persona_app(persona_key)` is called — this is cached via
   `@st.cache_resource` so each persona's LangGraph app is compiled once
4. The new persona's system prompt is used for all subsequent messages

---

## 8. File Structure

```
Spiritual_AI/
│
├── app.py                    ← Streamlit UI (persona selector + streaming chat)
├── geeta_chat.py             ← LangGraph pipeline + 5 personas + RAG logic
├── ingest.py                 ← PDF → ChromaDB ingestion pipeline
├── requirements.txt          ← Python dependencies
│
├── final_geeta.pdf           ← Bhagavad Gita (Sanskrit)
├── Patanjali-yogasutra.IGS.pdf  ← Yoga Sutras of Patanjali
├── The-Ten-Principal-Upanishads.pdf  ← Ten Principal Upanishads
│
├── chroma_db/                ← ChromaDB vector store (auto-created by ingest.py)
│   ├── chroma.sqlite3        ← Main database (14 MB)
│   └── <uuid>/               ← HNSW index files (vector search index)
│
├── chunks.json               ← All 2046 chunks saved for BM25 (auto-created)
│
├── .streamlit/
│   └── config.toml           ← Streamlit theme (warm dark spiritual theme)
│
├── .gitignore                ← Excludes spiritual_venv/, __pycache__/
└── TECHNICAL_DOCUMENT.md    ← This document
```

### Key file roles

**`geeta_chat.py`** — the brain
- Defines `PERSONAS` dict (6 persona system prompts)
- Defines `SpiritualState` (LangGraph state)
- Defines all 5 nodes: `guard`, `rewrite`, `retrieve`, `generate`, `reject`
- `build_app(vectorstore, embeddings, system_prompt)` → compiled LangGraph app
- `FastEmbeddings` — fastembed ONNX wrapper
- `BM25Retriever` — keyword search over `chunks.json`
- `compress()` — cosine similarity filter

**`app.py`** — the face
- `load_base()` — loads embeddings + ChromaDB once (cached)
- `load_persona_app(key)` — builds LangGraph app per persona (cached)
- Sidebar: persona radio buttons
- Chat loop: streams responses using `stream_mode=["updates","messages"]`
- Shows step indicators during the pipeline run

**`ingest.py`** — the librarian
- Auto-discovers all `*.pdf` files
- Extracts text, metadata (page, chapter, verse, source_text)
- Saves `chunks.json` for BM25
- Builds ChromaDB from scratch

---

## 9. Data Flow — End to End

```
User types: "I feel anxious and lost. How can I find peace?"
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Streamlit (app.py)            │
              │  Appends HumanMessage to       │
              │  session_state.messages        │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  LangGraph: guard node         │
              │                                │
              │  Prompt to Groq:               │
              │  "RELEVANT or IRRELEVANT?      │
              │   I feel anxious and lost..."  │
              │                                │
              │  Groq response: "RELEVANT"     │
              │  state.is_relevant = True      │
              └───────────────┬───────────────┘
                              │ (RELEVANT path)
                              ▼
              ┌───────────────────────────────┐
              │  LangGraph: rewrite node       │
              │                                │
              │  First question, no history    │
              │  → query = original question   │
              │  state.query = "anxious lost   │
              │  find peace Bhagavad Gita"     │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  LangGraph: retrieve node      │
              │                                │
              │  1. Vector search → 8 chunks   │
              │     (Gita Ch.2, Upanishads)    │
              │  2. BM25 search → 8 chunks     │
              │  3. Merge → 12 unique chunks   │
              │  4. Compress → 5 best chunks   │
              │                                │
              │  state.context = [5 passages]  │
              │  state.sources = ["Gita p.19", │
              │    "Upanishads p.43", ...]      │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  LangGraph: generate node      │
              │                                │
              │  System prompt (Healing Guru)  │
              │  + 5 retrieved passages        │
              │  + conversation history        │
              │  → Groq llama-3.3-70b          │
              │                                │
              │  Groq streams back:            │
              │  "The Gita speaks directly     │
              │   to your pain. In Chapter 2,  │
              │   Krishna says to Arjuna..."   │
              │                                │
              │  + "Sources: Gita p.19,        │
              │      Upanishads p.43"          │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Streamlit (app.py)            │
              │                                │
              │  Streams tokens to UI          │
              │  Shows step indicators         │
              │  Renders sources expander      │
              │  Updates session history       │
              └───────────────────────────────┘
```

---

## 10. API & Model Details

### Groq API

| Parameter | Value |
|---|---|
| Model | `llama-3.3-70b-versatile` |
| Temperature | `0.1` (low — consistent, factual) |
| Max tokens | `2048` per response |
| API key env var | `GROQ_API_KEY` |
| Speed | ~750 tokens/second (LPU hardware) |

The low temperature (0.1) is intentional — for scriptural Q&A, we want
consistent, grounded responses rather than creative variation.

### Embedding Model

| Parameter | Value |
|---|---|
| Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Provider | fastembed (ONNX Runtime — no GPU needed) |
| Dimensions | 384 |
| Languages | 50+ including Hindi/Sanskrit (Devanagari) |
| Download size | ~45 MB (cached after first run) |
| Normalisation | L2 normalised (cosine similarity ready) |

### ChromaDB

| Parameter | Value |
|---|---|
| Search type | MMR (Maximal Marginal Relevance) — diverse results |
| Fetch k | 20 candidates |
| Return k | 8 results |
| Persist directory | `./chroma_db/` |
| Index type | HNSW (approximate nearest neighbours) |

MMR is used instead of plain similarity search because it avoids returning
5 chunks that are nearly identical — it picks diverse passages covering
different aspects of the query.

---

## 11. Streamlit UI

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN AREA                        │
│                             │                                   │
│  🕉 Spiritual AI            │  🕉 General Guru                  │
│  AI Virtual Guru            │  All sacred texts · balanced      │
│                             │  ─────────────────────────────── │
│  Choose Your Guru:          │                                   │
│  ● 🕉 General Guru          │  [Chat messages appear here]      │
│  ○ 🙏 Bhakti Guru           │                                   │
│  ○ 🧘 Yoga Guru             │  🕉 (welcome message)             │
│  ○ 🔮 Meditation Guru       │                                   │
│  ○ ⚖️ Karma Guru            │  🧑 User message                  │
│  ○ 💚 Healing Guru          │                                   │
│                             │  🔍 Checking relevance…           │
│  Knowledge base:            │  ✍️ Understanding question…       │
│  • Bhagavad Gita            │  📖 Searching texts…             │
│  • Yoga Sutras              │  [Answer streams here...]         │
│  • Upanishads               │                                   │
│                             │  📖 Sources (expandable)          │
│  LLM: llama-3.3-70b         │                                   │
│  Search: Hybrid             │  ─────────────────────────────── │
│                             │  [Ask the Guru...]  [Send]        │
│  🗑 Clear conversation       │                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Streaming implementation

```python
# app.py uses LangGraph's dual stream mode:
for mode, data in app.stream(state, stream_mode=["updates", "messages"]):

    if mode == "updates":
        # Node completed — show which step just finished
        node = next(iter(data))
        # Update step indicator text

    elif mode == "messages":
        # LLM token received — stream to UI
        chunk, metadata = data
        if metadata["langgraph_node"] == "generate":
            full_content += chunk.content
            placeholder.markdown(full_content + "▌")  # ▌ = cursor
```

### Caching strategy

| Cache | Function | Reloaded when |
|---|---|---|
| `@st.cache_resource` | `load_base()` | App restarts |
| `@st.cache_resource(key=persona)` | `load_persona_app(key)` | New persona first time |
| `st.session_state` | `messages`, `chat_display` | Page refresh or Clear button |

Each persona's LangGraph graph is compiled once and cached. Switching
between personas is instant after the first use.

---

## 12. Setup & Run

### Prerequisites
- Python 3.10+
- Groq API key (free at console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shivamshyperlinkinfosystem-cloud/Spiritual_ai.git
cd Spiritual_ai

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the knowledge base (one-time)
python ingest.py
# Output: ✅ Ingestion complete — 2046 chunks from 3 texts

# 5. Run the app
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

### Adding new sacred texts
```bash
# Drop any PDF into the project folder
cp new_text.pdf /path/to/Spiritual_ai/

# Rebuild the knowledge base
python ingest.py
# Automatically discovers all *.pdf files
```

### Terminal mode (no UI)
```bash
export GROQ_API_KEY="gsk_..."
python geeta_chat.py
# Select a persona by number
# Chat in the terminal
```

---

## 13. Deployment on Streamlit Cloud

### Steps
1. Push code to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → connect `shivamshyperlinkinfosystem-cloud/Spiritual_ai`
4. Branch: `main` · File: `app.py`
5. **Settings → Secrets** → add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Deploy

### Why no ingestion needed on cloud?
The `chroma_db/` folder, `chunks.json`, and all PDFs are committed to the
GitHub repository. The cloud app loads the pre-built knowledge base directly.
No ingestion step needed on Streamlit Cloud.

### Cloud constraints
| Resource | Free Tier | Our Usage |
|---|---|---|
| RAM | 1 GB | ~400 MB (embeddings + ChromaDB) |
| CPU | Shared | ONNX embedding model — CPU-only |
| Disk | 1 GB | ~50 MB (chroma_db + chunks.json) |
| Timeout | None | N/A (persistent) |

---

## 14. Phase 1 Completion Status

| Requirement | Status | Notes |
|---|---|---|
| RAG infrastructure — vector DB + embedding pipeline | ✅ Done | ChromaDB + fastembed ONNX |
| RAG infrastructure — ingestion tooling | ✅ Done | `ingest.py` auto-discovers PDFs |
| AI Virtual Guru — 5 specialized personas | ✅ Done | Bhakti, Yoga, Meditation, Karma, Healing |
| Knowledge base — Bhagavad Gita | ✅ Done | 919 chunks |
| Knowledge base — Upanishads | ✅ Done | 349 chunks (Ten Principal) |
| Knowledge base — foundational texts | ✅ Done | Yoga Sutras — 778 chunks |
| Multi-turn conversational interface | ✅ Done | Full history in LangGraph state |
| Context retention | ✅ Done | `add_messages` reducer preserves history |
| Source grounding | ✅ Done | Ch/Verse citations + source text name |
| Topic guard — reject off-topic questions | ✅ Done | `guard` node with LLM classification |
| Query rewriting for follow-ups | ✅ Done | `rewrite` node |
| Hybrid search | ✅ Done | Vector MMR + BM25 combined |
| Contextual compression | ✅ Done | Cosine similarity filter |
| Streaming responses | ✅ Done | `stream_mode=["updates","messages"]` |
| Streamlit UI | ✅ Done | Persona selector + step indicators |
| GitHub repository | ✅ Done | Clean — source files only |
| Deployment-ready | ✅ Done | Streamlit Cloud compatible |
| Sub-2-second response | ⚠️ Partial | Answer streams within ~1-2s (Groq fast), but guard+rewrite add ~3-4s total |

---

## 15. What to Build Next (Phase 2)

### Reduce the 5-second wait time
The biggest UX issue. Three LLM calls run sequentially:
guard → rewrite → retrieve → generate.

**Options:**
- **Run guard + rewrite in parallel** (cut 1.5s)
- **Skip rewrite for first message** (already done for first turn)
- **Cache guard results** for repeated questions
- **Use a faster/smaller model** for guard + rewrite (e.g., Haiku 4.5)

### More sacred texts
Easy — drop PDF into folder and re-run `ingest.py`:
- Narada Bhakti Sutras (Bhakti persona)
- Vivekachudamani by Shankaracharya (Meditation persona)
- Hatha Yoga Pradipika (Yoga persona)
- Devi Mahatmyam (Healing persona)

### Voice interface
- User speaks → Whisper API transcribes → Gita answers → Text-to-speech reads it aloud
- Very fitting for a spiritual guru experience

### User accounts + conversation history
- Save conversations per user
- "Continue from last time"
- Personal spiritual journal

### Phase 2 persona additions
- **Dharmashastra Guru** — traditional law and ethics
- **Ayurveda Guru** — holistic health and healing texts
- **Tantra / Shakti Guru** — divine feminine and energy

---

*Document version 1.0 — Spiritual AI Phase 1*
*Built with LangGraph · Groq · ChromaDB · Streamlit*
