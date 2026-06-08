"""Tests for GATOR literature retrieval.

Tests query building, context formatting, and FAIL-system skipping.
Does NOT require a vector store — tests the logic in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gator.gate_engine import PhysicsProfile, load_config, load_records_from_csv, run_gates
from gator.literature_retrieval import (
    ExperimentalContext,
    RetrievedChunk,
    build_retrieval_query,
    format_rag_context,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "screening_config.yaml"
DATA_PATH = REPO_ROOT / "data" / "labels.csv"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def records():
    return load_records_from_csv(DATA_PATH)


@pytest.fixture
def profiles(config, records):
    return run_gates(config, records)


def _get_profile(profiles, system: str) -> PhysicsProfile:
    return next(p for p in profiles if p.system == system)


# ---------- Query building ----------


class TestQueryBuilding:
    def test_pristine_query_has_tio2(self, profiles, records):
        profile = _get_profile(profiles, "Pristine")
        record = next(r for r in records if r["System"] == "Pristine")
        query = build_retrieval_query("Pristine", profile, record)
        assert "TiO2" in query or "titanium dioxide" in query

    def test_mg_query_has_dopant(self, profiles, records):
        profile = _get_profile(profiles, "Mg-TiO2")
        record = next(r for r in records if r["System"] == "Mg-TiO2")
        query = build_retrieval_query("Mg-TiO2", profile, record)
        assert "Mg" in query

    def test_query_has_energy(self, profiles, records):
        profile = _get_profile(profiles, "Pristine")
        record = next(r for r in records if r["System"] == "Pristine")
        query = build_retrieval_query("Pristine", profile, record)
        assert "kJ/mol" in query

    def test_query_has_mode(self, profiles, records):
        profile = _get_profile(profiles, "Pristine")
        record = next(r for r in records if r["System"] == "Pristine")
        query = build_retrieval_query("Pristine", profile, record)
        assert "molecular" in query

    def test_query_has_temperature(self, profiles, records):
        profile = _get_profile(profiles, "Pristine")
        record = next(r for r in records if r["System"] == "Pristine")
        query = build_retrieval_query("Pristine", profile, record)
        assert "320" in query  # T50 at 1bar


# ---------- ExperimentalContext ----------


class TestExperimentalContext:
    def test_context_is_frozen(self):
        ctx = ExperimentalContext(system="test", chunks=(), query_used="query")
        with pytest.raises(AttributeError):
            ctx.system = "hacked"

    def test_chunk_is_frozen(self):
        chunk = RetrievedChunk(
            text="some text",
            title="Paper",
            doi="10.1234",
            source="semantic_scholar",
            relevance_score=0.9,
            chunk_index=0,
        )
        with pytest.raises(AttributeError):
            chunk.text = "hacked"


# ---------- RAG context formatting ----------


class TestFormatting:
    def test_empty_contexts(self):
        result = format_rag_context([])
        assert result == ""

    def test_contexts_with_no_chunks(self):
        ctx = ExperimentalContext(system="Pristine", chunks=(), query_used="q")
        result = format_rag_context([ctx])
        assert result == ""

    def test_formats_chunks_with_citations(self):
        chunk = RetrievedChunk(
            text="H2 desorption peak at 340 K on Mg-doped TiO2",
            title="Smith et al. 2024",
            doi="10.1234/test",
            source="semantic_scholar",
            relevance_score=0.85,
            chunk_index=0,
        )
        ctx = ExperimentalContext(
            system="Mg-TiO2",
            chunks=(chunk,),
            query_used="H2 adsorption Mg TiO2",
        )
        result = format_rag_context([ctx])
        assert "Mg-TiO2" in result
        assert "Smith et al. 2024" in result
        assert "10.1234/test" in result
        assert "340 K" in result

    def test_multiple_systems(self):
        chunk1 = RetrievedChunk(
            text="text1",
            title="Paper1",
            doi="doi1",
            source="ss",
            relevance_score=0.9,
            chunk_index=0,
        )
        chunk2 = RetrievedChunk(
            text="text2",
            title="Paper2",
            doi="doi2",
            source="ss",
            relevance_score=0.8,
            chunk_index=0,
        )
        ctxs = [
            ExperimentalContext(system="Pristine", chunks=(chunk1,), query_used="q1"),
            ExperimentalContext(system="Mg-TiO2", chunks=(chunk2,), query_used="q2"),
        ]
        result = format_rag_context(ctxs)
        assert "Pristine" in result
        assert "Mg-TiO2" in result


# ---------- Corpus builder (unit-testable parts) ----------


class TestCorpusBuilderUtils:
    def test_chunk_text(self):
        from gator.corpus_builder import chunk_text

        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        # Each chunk should have roughly chunk_size words
        assert len(chunks[0].split()) == 100

    def test_chunk_text_short(self):
        from gator.corpus_builder import chunk_text

        chunks = chunk_text("hello world", chunk_size=100)
        assert len(chunks) == 1

    def test_paper_candidate_frozen(self):
        from gator.corpus_builder import PaperCandidate

        p = PaperCandidate(
            title="Test",
            authors=["A"],
            year=2024,
            abstract="abs",
            doi=None,
            url=None,
            source="test",
            external_id="123",
        )
        with pytest.raises(AttributeError):
            p.title = "hacked"

    def test_manifest_roundtrip(self, tmp_path, monkeypatch):
        import gator.corpus_builder as cb
        from gator.corpus_builder import (
            load_approved_manifest,
            save_approved_manifest,
        )

        # Point manifest to tmp dir
        test_manifest = tmp_path / "approved.json"
        monkeypatch.setattr(cb, "MANIFEST_PATH", test_manifest)

        # Empty initially
        assert load_approved_manifest() == []

        # Save and reload
        data = [{"title": "Test Paper", "external_id": "123"}]
        save_approved_manifest(data)
        loaded = load_approved_manifest()
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test Paper"
