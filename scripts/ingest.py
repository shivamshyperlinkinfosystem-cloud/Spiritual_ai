"""
scripts/ingest.py — Build the knowledge base from all PDFs in data/pdfs/.

Run from the project root:
    python scripts/ingest.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

# ── Make src/ importable when run as a script ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDFS_DIR, VECTORSTORE, CHUNKS_PATH, EMBED_MODEL
from src.pipeline.embeddings import FastEmbeddings

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", "॥", "।", " ", ""]

# Sanskrit patterns
_VERSE_RE   = re.compile(r'H[§\s]?(\d+)H')
_CHAPTER_RE = re.compile(r'अध्याय\s*(\d+)', re.UNICODE)

# Filename → clean source name
_SOURCE_MAP = {
    "bhagavad":  "Bhagavad Gita",
    "geeta":     "Bhagavad Gita",
    "gita":      "Bhagavad Gita",
    "yoga":      "Yoga Sutras of Patanjali",
    "patanjali": "Yoga Sutras of Patanjali",
    "upanishad": "Ten Principal Upanishads",
}


def source_name(pdf_path: Path) -> str:
    stem = re.sub(r"[.\-_]", "", pdf_path.stem.lower())
    for key, name in _SOURCE_MAP.items():
        if key in stem:
            return name
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


def ingest() -> None:
    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌  No PDF files found in {PDFS_DIR}")
        sys.exit(1)

    print(f"📚  Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"     • {p.name}  →  \"{source_name(p)}\"")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    all_chunks: list[Document] = []

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

    # Save chunks.json for BM25
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾  Saving chunks.json → {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": c.page_content, "metadata": c.metadata} for c in all_chunks],
            f, ensure_ascii=False, indent=2,
        )

    # Build ChromaDB
    print(f"\n🔢  Loading embedding model: {EMBED_MODEL}")
    embeddings = FastEmbeddings(EMBED_MODEL)

    print(f"💾  Building ChromaDB → {VECTORSTORE}")
    if VECTORSTORE.exists():
        shutil.rmtree(VECTORSTORE)
    VECTORSTORE.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = None
    batch_size  = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embeddings,
                persist_directory=str(VECTORSTORE),
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, len(all_chunks))
        print(f"    Indexed {done}/{len(all_chunks)} chunks …", end="\r")

    print(f"\n\n✅  Done — {len(all_chunks)} chunks from {len(pdf_files)} texts")
    print("\n    Run the app:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
