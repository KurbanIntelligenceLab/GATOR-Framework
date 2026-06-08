"""Literature retrieval: physics-aware RAG for screening context.

Builds retrieval queries from PhysicsProfiles, queries the local vector
store, and packages results as ExperimentalContext objects for the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gate_engine import PhysicsProfile

__all__ = [
    "ExperimentalContext",
    "RetrievedChunk",
    "build_retrieval_query",
    "format_rag_context",
    "retrieve_literature",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved literature chunk with metadata."""

    text: str
    title: str
    doi: str
    source: str
    relevance_score: float
    chunk_index: int


@dataclass(frozen=True)
class ExperimentalContext:
    """Retrieved experimental context for one system."""

    system: str
    chunks: tuple[RetrievedChunk, ...]
    query_used: str


# ---------------------------------------------------------------------------
# Query construction (gate-informed)
# ---------------------------------------------------------------------------


def build_retrieval_query(
    system: str,
    profile: PhysicsProfile,
    record: dict[str, Any],
) -> str:
    """Build a physics-aware retrieval query from gate verdicts and descriptors.

    Gate verdicts shape the query: a system flagged "Outside--strong"
    needs literature on strong binding, not on enhancing uptake.
    """
    parts = [f"hydrogen H2 adsorption on {system}"]

    # Add substrate context
    if "TiO2" in system or system == "Pristine":
        parts.append("titanium dioxide TiO2 anatase nanoparticle")

    # Add dopant context
    for ae in ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]:
        if ae in system:
            parts.append(f"{ae} doped modified")
            break

    # Add energetic context from record
    e_ads = record.get("E_ads_kJ_mol")
    if e_ads is not None:
        parts.append(f"adsorption energy approximately {abs(e_ads):.0f} kJ/mol")

    # Add mode context from gate verdicts
    for v in profile.verdicts:
        if v.gate_name == "adsorption_mode":
            parts.append(f"{v.physics_label} adsorption mode")
        elif v.gate_name == "deliverability":
            if "under-load" in v.physics_label:
                parts.append("weak physisorption low coverage")
            elif "heating-needed" in v.physics_label:
                parts.append("strong chemisorption high desorption temperature")
            else:
                parts.append("near-ambient reversible storage")

    # Add T50 context
    t50 = record.get("T50_at_1bar_K")
    if t50 is not None:
        parts.append(f"desorption temperature {t50:.0f} K")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_literature(
    profiles: Sequence[PhysicsProfile],
    records: Sequence[dict[str, Any]],
    top_k: int = 5,
    index_dir: str = "literature/index",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> tuple[ExperimentalContext, ...]:
    """Retrieve experimental literature for each system.

    FAIL systems are skipped — no point retrieving literature for
    excluded candidates.
    """
    index_path = Path(index_dir)
    if not index_path.exists():
        return ()

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return ()

    client = chromadb.PersistentClient(path=str(index_path))

    try:
        collection = client.get_collection("gator_literature")
    except Exception:
        return ()

    model = SentenceTransformer(embedding_model)

    # Build system→record lookup
    record_map = {r.get("System", ""): r for r in records}

    contexts: list[ExperimentalContext] = []

    for profile in profiles:
        # Skip FAIL systems — no point retrieving literature for them
        if profile.overall == "fail":
            contexts.append(
                ExperimentalContext(
                    system=profile.system,
                    chunks=(),
                    query_used="(skipped — system classified as FAIL)",
                )
            )
            continue

        record = record_map.get(profile.system, {})
        query = build_retrieval_query(profile.system, profile, record)

        # Embed query and search
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    title=meta.get("title", ""),
                    doi=meta.get("doi", ""),
                    source=meta.get("source", ""),
                    relevance_score=1.0 - dist,  # Convert distance to similarity
                    chunk_index=meta.get("chunk_index", 0),
                )
            )

        contexts.append(
            ExperimentalContext(
                system=profile.system,
                chunks=tuple(chunks),
                query_used=query,
            )
        )

    return tuple(contexts)


# ---------------------------------------------------------------------------
# Formatting for LLM prompt
# ---------------------------------------------------------------------------


def _truncate_to_sentence(text: str, max_chars: int = 150) -> str:
    """Truncate text to the last complete sentence within max_chars."""
    if len(text) <= max_chars:
        return text.strip()
    snippet = text[:max_chars]
    # Try to cut at last sentence boundary
    for sep in (". ", "? ", "! "):
        last = snippet.rfind(sep)
        if last > 40:
            return snippet[: last + 1].strip()
    # Fall back to last comma or semicolon
    for sep in (", ", "; "):
        last = snippet.rfind(sep)
        if last > 40:
            return snippet[:last].strip()
    return snippet.strip() + "..."


def format_rag_context(contexts: Sequence[ExperimentalContext]) -> str:
    """Format retrieved literature as compact numbered references.

    Each chunk becomes a one-line bullet with [N], title, DOI, and
    a truncated key finding. A bibliography is appended at the end.
    """
    if not contexts or all(len(c.chunks) == 0 for c in contexts):
        return ""

    sections = []
    ref_list: list[str] = []
    seen_keys: dict[tuple[str, str], int] = {}  # (title, doi) → ref number

    for ctx in contexts:
        if not ctx.chunks:
            continue

        lines = [f"### {ctx.system}"]
        for chunk in ctx.chunks:
            # Assign or reuse a reference number
            ref_key = (chunk.title.strip().lower(), (chunk.doi or "").strip().lower())
            if ref_key not in seen_keys:
                ref_num = len(ref_list) + 1
                seen_keys[ref_key] = ref_num
                citation = f"[{ref_num}] {chunk.title}"
                if chunk.doi:
                    citation += f", DOI: {chunk.doi}"
                ref_list.append(citation)
            else:
                ref_num = seen_keys[ref_key]

            summary = _truncate_to_sentence(chunk.text)
            lines.append(f"[{ref_num}] {chunk.title}")
            lines.append(f"  → {summary}")

        sections.append("\n".join(lines))

    # Append bibliography
    if ref_list:
        sections.append("### References\n" + "\n".join(ref_list))

    return "\n\n".join(sections)
