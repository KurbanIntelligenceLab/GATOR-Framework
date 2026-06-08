"""Corpus builder: semi-automated literature acquisition.

Searches Semantic Scholar and arXiv for relevant papers, presents
candidates for user approval, downloads PDFs, extracts text, chunks
by section, and indexes into a local vector store.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PaperCandidate",
    "approve_papers",
    "auto_build_corpus",
    "chunk_text",
    "extract_text_from_pdf",
    "index_corpus",
    "load_approved_manifest",
    "save_approved_manifest",
    "search_arxiv",
    "search_literature",
    "search_openalex",
    "search_semantic_scholar",
]

LITERATURE_DIR = Path("literature")
PAPERS_DIR = LITERATURE_DIR / "papers"
INDEX_DIR = LITERATURE_DIR / "index"
MANIFEST_PATH = LITERATURE_DIR / "approved.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperCandidate:
    """A paper candidate from search results."""

    title: str
    authors: list[str]
    year: int | None
    abstract: str
    doi: str | None
    url: str | None
    source: str  # "semantic_scholar" or "arxiv"
    external_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "url": self.url,
            "source": self.source,
            "external_id": self.external_id,
        }


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------


def search_semantic_scholar(query: str, limit: int = 10) -> list[PaperCandidate]:
    """Search Semantic Scholar for papers matching the query."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded}&limit={limit}"
        f"&fields=title,authors,year,abstract,externalIds,url"
    )

    headers = {"Accept": "application/json"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Semantic Scholar API error: {e}")
        return []

    papers = data.get("data", [])
    candidates = []

    for p in papers:
        ext_ids = p.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        authors = [a.get("name", "") for a in (p.get("authors") or [])]

        candidates.append(
            PaperCandidate(
                title=p.get("title", ""),
                authors=authors,
                year=p.get("year"),
                abstract=p.get("abstract") or "",
                doi=doi,
                url=p.get("url"),
                source="semantic_scholar",
                external_id=p.get("paperId", ""),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------


def search_arxiv(query: str, limit: int = 10) -> list[PaperCandidate]:
    """Search arXiv for papers matching the query."""
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={limit}"
    )

    req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"arXiv API error: {e}")
        return []

    # Simple XML parsing without external dependency
    candidates = []
    entries = raw.split("<entry>")[1:]  # Skip the feed header

    for entry in entries[:limit]:
        title = _extract_xml(entry, "title").strip().replace("\n", " ")
        abstract = _extract_xml(entry, "summary").strip().replace("\n", " ")
        arxiv_id = _extract_xml(entry, "id").strip()

        # Extract authors
        authors = []
        for author_block in entry.split("<author>")[1:]:
            name = _extract_xml(author_block, "name")
            if name:
                authors.append(name.strip())

        # Extract DOI if present
        doi = None
        if "doi.org/" in entry:
            doi_start = entry.find("doi.org/") + 8
            doi_end = entry.find('"', doi_start)
            if doi_end > doi_start:
                doi = entry[doi_start:doi_end]

        candidates.append(
            PaperCandidate(
                title=title,
                authors=authors,
                year=None,
                abstract=abstract,
                doi=doi,
                url=arxiv_id,
                source="arxiv",
                external_id=arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# OpenAlex API
# ---------------------------------------------------------------------------


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def search_openalex(query: str, limit: int = 10) -> list[PaperCandidate]:
    """Search OpenAlex for papers matching the query."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works"
        f"?filter=default.search:{encoded}"
        f"&per_page={limit}"
        f"&select=id,title,authorships,publication_year,abstract_inverted_index,doi"
    )

    headers = {"Accept": "application/json"}
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"OpenAlex API error: {e}")
        return []

    results = data.get("results", [])
    candidates = []

    for work in results:
        title = work.get("title") or ""
        authors = [
            a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])
        ]
        year = work.get("publication_year")
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})
        doi_raw = work.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
        openalex_id = work.get("id") or ""

        candidates.append(
            PaperCandidate(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                doi=doi,
                url=openalex_id,
                source="openalex",
                external_id=openalex_id,
            )
        )

    return candidates


def _extract_xml(text: str, tag: str) -> str:
    """Extract text content between XML tags (simple, no-dependency parser)."""
    start_tag = f"<{tag}"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start < 0:
        return ""
    # Skip past the opening tag (including attributes)
    content_start = text.find(">", start) + 1
    end = text.find(end_tag, content_start)
    if end < 0:
        return ""
    return text[content_start:end]


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------


def load_approved_manifest() -> list[dict[str, Any]]:
    """Load the approved papers manifest."""
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open() as f:
        data: list[dict[str, Any]] = json.load(f)
        return data


def save_approved_manifest(manifest: list[dict[str, Any]]) -> None:
    """Save the approved papers manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# PDF extraction and chunking
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "PDF extraction requires PyMuPDF.\nInstall: `pip install gator[rag]`"
        ) from e

    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# CLI-facing functions
# ---------------------------------------------------------------------------


def search_literature(query: str, limit: int = 10) -> None:
    """Search Semantic Scholar, arXiv, and OpenAlex, display results for review."""
    print(f"\nSearching for: {query!r}\n")

    print("--- Semantic Scholar ---")
    ss_results = search_semantic_scholar(query, limit=limit)
    for i, p in enumerate(ss_results):
        print(f"\n[SS-{i + 1}] {p.title}")
        print(f"  Authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"  Year: {p.year}  DOI: {p.doi or 'N/A'}")
        abstract_preview = p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract
        print(f"  Abstract: {abstract_preview}")

    print("\n--- arXiv ---")
    arxiv_results = search_arxiv(query, limit=limit)
    for i, p in enumerate(arxiv_results):
        print(f"\n[AX-{i + 1}] {p.title}")
        print(f"  Authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"  URL: {p.url}")
        abstract_preview = p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract
        print(f"  Abstract: {abstract_preview}")

    print("\n--- OpenAlex ---")
    oa_results = search_openalex(query, limit=limit)
    for i, p in enumerate(oa_results):
        print(f"\n[OA-{i + 1}] {p.title}")
        print(f"  Authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"  Year: {p.year}  DOI: {p.doi or 'N/A'}")
        abstract_preview = p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract
        print(f"  Abstract: {abstract_preview}")

    # Save candidates for approval
    all_candidates = [c.to_dict() for c in ss_results + arxiv_results + oa_results]
    candidates_path = LITERATURE_DIR / "candidates.json"
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with candidates_path.open("w") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"\n{len(all_candidates)} candidates saved to {candidates_path}")
    print("Run `python -m gator corpus approve` to select papers.")


def approve_papers() -> None:
    """Interactive approval of paper candidates."""
    candidates_path = LITERATURE_DIR / "candidates.json"
    if not candidates_path.exists():
        print("No candidates found. Run `python -m gator corpus search` first.")
        return

    with candidates_path.open() as f:
        candidates = json.load(f)

    manifest = load_approved_manifest()
    approved_ids = {p["external_id"] for p in manifest}

    print(f"\n{len(candidates)} candidates available, {len(manifest)} already approved.\n")

    for i, c in enumerate(candidates):
        if c["external_id"] in approved_ids:
            print(f"[{i + 1}] SKIP (already approved): {c['title'][:80]}")
            continue

        print(f"\n[{i + 1}/{len(candidates)}] {c['title']}")
        print(f"  Source: {c['source']}  Year: {c.get('year', 'N/A')}")
        abstract_preview = c.get("abstract", "")[:300]
        print(f"  Abstract: {abstract_preview}")

        response = input("  Approve? (y/n/q): ").strip().lower()
        if response == "q":
            break
        if response == "y":
            manifest.append(c)
            approved_ids.add(c["external_id"])
            print("  ✓ Approved")

    save_approved_manifest(manifest)
    print(f"\n{len(manifest)} papers in approved manifest.")


def _deduplicate_candidates(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Deduplicate paper candidates by DOI or normalized title."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[PaperCandidate] = []

    for c in candidates:
        # Check DOI first
        if c.doi:
            doi_norm = c.doi.strip().lower()
            if doi_norm in seen_dois:
                continue
            seen_dois.add(doi_norm)

        # Fall back to normalized title
        title_norm = c.title.strip().lower()
        if title_norm in seen_titles:
            continue
        seen_titles.add(title_norm)

        unique.append(c)

    return unique


_DEFAULT_QUERIES = [
    "hydrogen adsorption TiO2 nanoparticle",
    "H2 storage titanium dioxide anatase nanocluster",
    "alkaline earth doped TiO2 hydrogen",
    "reversible hydrogen storage metal oxide nanoparticle",
    "DFTB hydrogen adsorption energy metal oxide",
]


def auto_build_corpus(
    queries: list[str] | None = None,
    limit: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    """Search, auto-approve, and index literature in one step.

    Searches Semantic Scholar, arXiv, and OpenAlex with domain-relevant
    queries, deduplicates results, saves them as approved, and indexes
    abstracts into the ChromaDB vector store.
    """
    queries = queries or _DEFAULT_QUERIES
    all_candidates: list[PaperCandidate] = []

    for query in queries:
        print(f"\nSearching: {query!r}")
        ss = search_semantic_scholar(query, limit=limit)
        print(f"  Semantic Scholar: {len(ss)} results")
        ax = search_arxiv(query, limit=limit)
        print(f"  arXiv: {len(ax)} results")
        oa = search_openalex(query, limit=limit)
        print(f"  OpenAlex: {len(oa)} results")
        all_candidates.extend(ss + ax + oa)

    # Deduplicate
    unique = _deduplicate_candidates(all_candidates)
    print(f"\nTotal: {len(all_candidates)} results, {len(unique)} unique after dedup")

    # Filter out papers with no abstract (useless for RAG)
    with_abstract = [c for c in unique if c.abstract.strip()]
    print(f"Papers with abstracts: {len(with_abstract)}")

    if not with_abstract:
        print("No papers with abstracts found. Aborting.")
        return

    # Auto-approve: merge with existing manifest
    manifest = load_approved_manifest()
    existing_ids = {p["external_id"] for p in manifest}
    new_count = 0
    for c in with_abstract:
        if c.external_id not in existing_ids:
            manifest.append(c.to_dict())
            existing_ids.add(c.external_id)
            new_count += 1
    save_approved_manifest(manifest)
    print(f"Approved: {new_count} new papers ({len(manifest)} total in manifest)")

    # Index into ChromaDB
    print(f"\nIndexing with embedding model '{embedding_model}'...")
    index_corpus(embedding_model=embedding_model)


def index_corpus(embedding_model: str = "all-MiniLM-L6-v2") -> None:
    """Index approved papers into the vector store."""
    manifest = load_approved_manifest()
    if not manifest:
        print("No approved papers. Run search and approve first.")
        return

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Corpus indexing requires sentence-transformers and chromadb.\n"
            "Install: `pip install gator[rag]`"
        ) from e

    print(f"Indexing {len(manifest)} approved papers with model '{embedding_model}'...")

    # Initialize embedding model and vector store
    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_or_create_collection("gator_literature")

    indexed = 0
    for paper in manifest:
        # Check if PDF exists locally
        paper_id = paper["external_id"].replace("/", "_")
        pdf_path = PAPERS_DIR / f"{paper_id}.pdf"

        if pdf_path.exists():
            try:
                text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(f"  Error extracting {pdf_path.name}: {e}")
                continue
        else:
            # Use abstract as fallback
            text = paper.get("abstract", "")
            if not text:
                continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = model.encode(chunks).tolist()
        ids = [f"{paper_id}_chunk_{j}" for j in range(len(chunks))]
        metadatas = [
            {
                "title": paper["title"],
                "doi": paper.get("doi") or "",
                "source": paper["source"],
                "chunk_index": j,
            }
            for j in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        indexed += 1
        print(f"  Indexed: {paper['title'][:60]}... ({len(chunks)} chunks)")

    print(f"\nDone. {indexed} papers indexed in {INDEX_DIR}")
