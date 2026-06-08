"""Tests for corpus builder: OpenAlex search, deduplication, auto-corpus."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gator.corpus_builder import (
    PaperCandidate,
    _deduplicate_candidates,
    _reconstruct_abstract,
    search_openalex,
)


class TestReconstructAbstract:
    """Test OpenAlex abstract_inverted_index reconstruction."""

    def test_basic_reconstruction(self):
        inverted = {"hydrogen": [0], "adsorption": [1], "on": [2], "TiO2": [3]}
        assert _reconstruct_abstract(inverted) == "hydrogen adsorption on TiO2"

    def test_repeated_words(self):
        inverted = {"the": [0, 4], "cat": [1], "sat": [2], "on": [3], "mat": [5]}
        assert _reconstruct_abstract(inverted) == "the cat sat on the mat"

    def test_empty_index(self):
        assert _reconstruct_abstract({}) == ""

    def test_none_index(self):
        assert _reconstruct_abstract(None) == ""


class TestSearchOpenalex:
    """Test OpenAlex API search with mocked HTTP."""

    def _mock_response(self, results: list) -> MagicMock:
        body = json.dumps({"results": results}).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("urllib.request.urlopen")
    def test_parses_results(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response(
            [
                {
                    "id": "https://openalex.org/W12345",
                    "title": "Hydrogen Storage on TiO2",
                    "authorships": [
                        {"author": {"display_name": "Alice Smith"}},
                        {"author": {"display_name": "Bob Jones"}},
                    ],
                    "publication_year": 2023,
                    "abstract_inverted_index": {"hydrogen": [0], "storage": [1]},
                    "doi": "https://doi.org/10.1234/test",
                }
            ]
        )

        results = search_openalex("hydrogen TiO2", limit=5)
        assert len(results) == 1
        p = results[0]
        assert p.title == "Hydrogen Storage on TiO2"
        assert p.authors == ["Alice Smith", "Bob Jones"]
        assert p.year == 2023
        assert p.abstract == "hydrogen storage"
        assert p.doi == "10.1234/test"
        assert p.source == "openalex"

    @patch("urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response([])
        results = search_openalex("nothing here")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_missing_fields_handled(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response(
            [
                {
                    "id": "https://openalex.org/W99999",
                    "title": "No Details Paper",
                    "authorships": None,
                    "publication_year": None,
                    "abstract_inverted_index": None,
                    "doi": None,
                }
            ]
        )
        results = search_openalex("test")
        assert len(results) == 1
        assert results[0].authors == []
        assert results[0].abstract == ""
        assert results[0].doi is None


class TestDeduplication:
    """Test paper candidate deduplication."""

    def _paper(self, title: str, doi: str | None = None, ext_id: str = "") -> PaperCandidate:
        return PaperCandidate(
            title=title,
            authors=[],
            year=2024,
            abstract="",
            doi=doi,
            url=None,
            source="test",
            external_id=ext_id or title,
        )

    def test_dedup_by_doi(self):
        papers = [
            self._paper("Paper A", doi="10.1234/a", ext_id="1"),
            self._paper("Paper B", doi="10.1234/a", ext_id="2"),  # Same DOI
        ]
        unique = _deduplicate_candidates(papers)
        assert len(unique) == 1
        assert unique[0].title == "Paper A"

    def test_dedup_by_title(self):
        papers = [
            self._paper("Hydrogen Storage on TiO2", ext_id="1"),
            self._paper("Hydrogen Storage on TiO2", ext_id="2"),  # Same title
        ]
        unique = _deduplicate_candidates(papers)
        assert len(unique) == 1

    def test_different_papers_kept(self):
        papers = [
            self._paper("Paper A", doi="10.1234/a"),
            self._paper("Paper B", doi="10.1234/b"),
        ]
        unique = _deduplicate_candidates(papers)
        assert len(unique) == 2

    def test_empty_input(self):
        assert _deduplicate_candidates([]) == []

    def test_case_insensitive_doi(self):
        papers = [
            self._paper("Paper A", doi="10.1234/ABC", ext_id="1"),
            self._paper("Paper B", doi="10.1234/abc", ext_id="2"),
        ]
        unique = _deduplicate_candidates(papers)
        assert len(unique) == 1

    def test_case_insensitive_title(self):
        papers = [
            self._paper("Hydrogen Storage", ext_id="1"),
            self._paper("hydrogen storage", ext_id="2"),
        ]
        unique = _deduplicate_candidates(papers)
        assert len(unique) == 1

    def test_immutability(self):
        p = self._paper("Test")
        with pytest.raises(AttributeError):
            p.title = "Modified"
