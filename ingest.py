"""
Spiritual AI — ingestion pipeline.
Auto-discovers every *.pdf in the project folder, tags each chunk with
the source text name, and stores in ChromaDB + chunks.json for BM25.

Run once (or whenever you add new PDFs):
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
BASE_DIR    = Path(__file__).parent
CHROMA_DIR  = BASE_DIR / "chroma_db"
CHUNKS_PATH = BASE_DIR / "chunks.json"

EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", "॥", "।", " ", ""]

# Sanskrit verse pattern: H§47H, H 13 H
_VERSE_RE   = re.compile(r'H[§\s]?(\d+)H')
_CHAPTER_RE = re.compile(r'अध्याय\s*(\d+)', re.UNICODE)

# Keyword → clean source name mapping
_SOURCE_MAP = {
    "geeta":      "Bhagavad Gita",
    "gita":       "Bhagavad Gita",
    "yogasutra":  "Yoga Sutras of Patanjali",
    "yoga_sutra": "Yoga Sutras of Patanjali",
    "patanjali":  "Yoga Sutras of Patanjali",
    "upanishad":  "Ten Principal Upanishads",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
class FastEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


def source_name(pdf_path: Path) -> str:
    """Derive a clean human-readable source name from a PDF filename."""
    # Normalise: lowercase, strip extension, remove dots/dashes/underscores
    stem = re.sub(r"[.\-_]", "", pdf_path.stem.lower())
    for key, name in _SOURCE_MAP.items():
        if key.replace("_", "") in stem:
            return name
    # Fallback: prettify the filename
    return re.sub(r"[-_.]", " ", pdf_path.stem).strip().title()


def extract_meta(text: str, page_idx: int, src: str) -> dict:
    meta: dict = {"page": page_idx, "source": src, "source_text": src}
    verses  = _VERSE_RE.findall(text)
    chapter = _CHAPTER_RE.search(text)
    if verses:
        meta["verse"]   = verses[-1]
    if chapter:
        meta["chapter"] = chapter.group(1)
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest() -> None:
    pdf_files = sorted(BASE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌  No PDF files found in", BASE_DIR)
        sys.exit(1)

    print(f"📚  Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"     • {p.name}  →  \"{source_name(p)}\"")

    all_chunks: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    for pdf_path in pdf_files:
        src = source_name(pdf_path)
        print(f"\n📖  Loading «{src}» ({pdf_path.stat().st_size // 1024} KB) …")

        reader = pypdf.PdfReader(str(pdf_path))
        pages  = [
            Document(
                page_content=page.extract_text() or "",
                metadata=extract_meta(page.extract_text() or "", i, src),
            )
            for i, page in enumerate(reader.pages)
        ]
        pages  = [p for p in pages if p.page_content.strip()]
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"     {len(pages)} pages → {len(chunks)} chunks")

    print(f"\n📦  Total: {len(all_chunks)} chunks from {len(pdf_files)} texts")

    # ── Save chunks.json for BM25 ─────────────────────────────────────────────
    print(f"💾  Saving chunks.json …")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": c.page_content, "metadata": c.metadata} for c in all_chunks],
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
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, len(all_chunks))
        print(f"    Indexed {done}/{len(all_chunks)} chunks …", end="\r")

    print(f"\n\n✅  Ingestion complete — {len(all_chunks)} chunks from {len(pdf_files)} texts")
    sources = sorted({source_name(p) for p in pdf_files})
    print("    Texts in knowledge base:")
    for s in sources:
        print(f"     • {s}")
    print("\n    You can now run:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
