"""
Bhagavad Gita PDF ingestion pipeline.

Improvements:
  - Extracts verse numbers from Devanagari text as metadata
  - Saves chunks.json alongside chroma_db for BM25 hybrid search

Run once:
    python ingest.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pypdf
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PDF_PATH   = BASE_DIR / "final_geeta.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHUNKS_PATH = BASE_DIR / "chunks.json"   # saved for BM25 retriever

EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", "॥", "।", " ", ""]

# Verse pattern in the PDF: H§47H  H§2H  H 13 H etc.
_VERSE_RE   = re.compile(r'H[§\s]?(\d+)H')
# Adhyaya (chapter) marker in Hindi: "अध्याय 2" or similar
_CHAPTER_RE = re.compile(r'अध्याय\s*(\d+)', re.UNICODE)


# ── fastembed wrapper ─────────────────────────────────────────────────────────
class FastEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


def _extract_meta(text: str, page_idx: int) -> dict:
    """Extract chapter and verse numbers from page text."""
    meta: dict = {"page": page_idx, "source": "Bhagavad Gita"}
    verses  = _VERSE_RE.findall(text)
    chapter = _CHAPTER_RE.search(text)
    if verses:
        meta["verse"] = verses[-1]          # last verse number on page
    if chapter:
        meta["chapter"] = chapter.group(1)
    return meta


def ingest() -> None:
    if not PDF_PATH.exists():
        print(f"❌  PDF not found: {PDF_PATH}")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"📖  Loading {PDF_PATH.name}  ({PDF_PATH.stat().st_size // 1024} KB) …")
    reader = pypdf.PdfReader(str(PDF_PATH))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata=_extract_meta(page.extract_text() or "", i),
        )
        for i, page in enumerate(reader.pages)
    ]
    pages = [p for p in pages if p.page_content.strip()]
    print(f"    Loaded {len(pages)} pages")

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(pages)
    print(f"    Split into {len(chunks)} chunks")

    # ── Save chunks for BM25 ──────────────────────────────────────────────────
    print(f"💾  Saving {len(chunks)} chunks to {CHUNKS_PATH.name} for BM25 …")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": c.page_content, "metadata": c.metadata} for c in chunks],
            f, ensure_ascii=False, indent=2,
        )

    # ── Embed & store in ChromaDB ─────────────────────────────────────────────
    print(f"\n🔢  Loading embedding model: {EMBED_MODEL}")
    embeddings = FastEmbeddings(EMBED_MODEL)

    print(f"💾  Building ChromaDB at {CHROMA_DIR} …")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = None
    batch_size  = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, len(chunks))
        print(f"    Indexed {done}/{len(chunks)} chunks …", end="\r")

    print(f"\n\n✅  Ingestion complete — {len(chunks)} chunks in {CHROMA_DIR}")
    print("    You can now run:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
