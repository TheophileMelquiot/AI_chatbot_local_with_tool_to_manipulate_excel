"""
tests/test_detect_values.py
===========================
Unit tests for detecteur_autonomous.detect_values()
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detecteur_autonomous import _best_column_match, _fuzzy_ratio, _normalize, detect_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_excel(tmp_path, data: dict, sheet_name: str = "Sheet1") -> str:
    """Write a small DataFrame to an .xlsx file and return its path."""
    df = pd.DataFrame(data)
    path = str(tmp_path / "test_data.xlsx")
    df.to_excel(path, index=False, sheet_name=sheet_name)
    return path


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lower_case(self):
        assert _normalize("ABC") == "abc"

    def test_strip_accents(self):
        assert _normalize("Identifiant de Champ") == "identifiant de champ"
        assert _normalize("montant") == "montant"
        assert _normalize("téléphone") == "telephone"

    def test_collapse_whitespace(self):
        assert _normalize("  id  client ") == "id client"


# ---------------------------------------------------------------------------
# _fuzzy_ratio
# ---------------------------------------------------------------------------

class TestFuzzyRatio:
    def test_identical(self):
        assert _fuzzy_ratio("Identifiant de Champ", "Identifiant de Champ") == pytest.approx(1.0)

    def test_case_insensitive(self):
        ratio = _fuzzy_ratio("identifiant de champ", "Identifiant de Champ")
        assert ratio == pytest.approx(1.0)

    def test_partial_match(self):
        ratio = _fuzzy_ratio("id champ", "Identifiant de Champ")
        assert 0.0 < ratio < 1.0

    def test_no_match(self):
        ratio = _fuzzy_ratio("xyz123", "Identifiant de Champ")
        assert ratio < 0.5


# ---------------------------------------------------------------------------
# _best_column_match
# ---------------------------------------------------------------------------

class TestBestColumnMatch:
    COLS = ["Identifiant de Champ", "Montant EUR", "Date Opération", "Statut"]

    def test_exact_match(self):
        assert _best_column_match(self.COLS, "Identifiant de Champ") == "Identifiant de Champ"

    def test_case_insensitive_exact(self):
        assert _best_column_match(self.COLS, "identifiant de champ") == "Identifiant de Champ"

    def test_accent_insensitive(self):
        assert _best_column_match(self.COLS, "Date Operation") == "Date Opération"

    def test_no_match_returns_none(self):
        assert _best_column_match(self.COLS, "ZZZZNOTEXIST", threshold=0.9) is None

    def test_threshold_filtering(self):
        # Force a very high threshold so partial matches are rejected
        result = _best_column_match(self.COLS, "Montant", threshold=0.99)
        assert result is None


# ---------------------------------------------------------------------------
# detect_values
# ---------------------------------------------------------------------------

class TestDetectValues:
    def test_exact_column_hint_finds_rows(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {
                "Identifiant de Champ": ["C8-01CE03", "C8-XXXXXX", "C8-01CE03"],
                "Montant": [100, 200, 300],
            },
        )
        results = detect_values(excel_path, "C8-01CE03", column_hint="Identifiant de Champ")
        assert len(results) == 2
        for r in results:
            assert r["column"] == "Identifiant de Champ"
            assert str(r["matched_value"]) == "C8-01CE03"

    def test_fuzzy_column_hint(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {
                "Identifiant de Champ": ["VAL1", "VAL2"],
                "Montant": [1, 2],
            },
        )
        # Provide a slightly different hint (accent-free)
        results = detect_values(excel_path, "VAL1", column_hint="Identifiant de champ")
        assert len(results) == 1
        assert results[0]["matched_value"] == "VAL1"

    def test_no_column_hint_searches_all(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {
                "Col A": ["foo", "bar"],
                "Col B": ["baz", "foo"],
            },
        )
        results = detect_values(excel_path, "foo")
        assert len(results) == 2
        matched_cols = {r["column"] for r in results}
        assert matched_cols == {"Col A", "Col B"}

    def test_no_results_when_value_absent(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {"Col": ["a", "b", "c"]},
        )
        results = detect_values(excel_path, "NOTEXIST", column_hint="Col")
        assert results == []

    def test_case_insensitive_by_default(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {"Col": ["Hello", "HELLO", "world"]},
        )
        results = detect_values(excel_path, "hello", column_hint="Col")
        assert len(results) == 2

    def test_case_sensitive_flag(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {"Col": ["Hello", "HELLO", "hello"]},
        )
        results = detect_values(excel_path, "hello", column_hint="Col", case_sensitive=True)
        assert len(results) == 1
        assert results[0]["matched_value"] == "hello"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_values(str(tmp_path / "nonexistent.xlsx"), "value")

    def test_column_hint_not_found_raises(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {"Col A": [1, 2]},
        )
        with pytest.raises(ValueError, match="did not match any column"):
            detect_values(excel_path, "1", column_hint="ZZZZNOTEXIST", fuzzy_threshold=0.99)

    def test_result_metadata(self, tmp_path):
        excel_path = _make_excel(
            tmp_path,
            {"ID": ["X1"], "Name": ["Alice"]},
        )
        results = detect_values(excel_path, "X1", column_hint="ID")
        assert len(results) == 1
        r = results[0]
        assert "sheet" in r
        assert "row_index" in r
        assert "excel_row" in r
        assert "column" in r
        assert "matched_value" in r
        assert "row_data" in r
        assert r["excel_row"] == r["row_index"] + 2

    def test_excel_row_offset(self, tmp_path):
        """excel_row should be row_index + 2 (header row + 1-based indexing)."""
        excel_path = _make_excel(
            tmp_path,
            {"Col": ["a", "b", "target", "d"]},
        )
        results = detect_values(excel_path, "target", column_hint="Col")
        assert len(results) == 1
        r = results[0]
        assert r["row_index"] == 2          # 0-based DataFrame index (3rd data row)
        assert r["excel_row"] == 4          # Excel row 1=header, row 2/3/4=data rows
