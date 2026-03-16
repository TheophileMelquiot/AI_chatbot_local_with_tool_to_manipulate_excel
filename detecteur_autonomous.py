"""
detecteur_autonomous.py
=======================
Provides detect_values() to search for a value in an Excel file.

When a column_hint is given, the matching column is found by fuzzy
string comparison of the actual column names against the hint – the
ML classifier is intentionally NOT used in that path because the
classifier maps headers to semantic categories (id_client, montant …)
whereas column_hint supplies the literal Excel column name.

Usage
-----
    from detecteur_autonomous import detect_values

    results = detect_values(
        "CARTO3_virements_S2_2025.xlsx",
        "C8-01CE03",
        column_hint="Identifiant de Champ",
    )
    for row in results:
        print(row)
"""

import logging
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [detecteur_autonomous] %(message)s",
)

# Minimum fuzzy-match ratio (0–1) to accept a column as matching the hint.
FUZZY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip accents and collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(without_accents.lower().split())


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_column_match(
    columns: List[str],
    hint: str,
    threshold: float = FUZZY_THRESHOLD,
) -> Optional[str]:
    """
    Return the column name that best matches *hint* via fuzzy comparison.

    Steps:
    1. Exact match (after normalisation).
    2. get_close_matches from difflib.
    3. Best SequenceMatcher ratio above *threshold*.

    Returns None when no column meets the threshold.
    """
    hint_norm = _normalize(hint)
    norms = {col: _normalize(col) for col in columns}

    # 1. Exact match after normalisation
    for col, norm in norms.items():
        if norm == hint_norm:
            logger.info("Exact match found: %r", col)
            return col

    # 2. difflib.get_close_matches (fast pre-filter)
    norm_list = list(norms.values())
    close = get_close_matches(hint_norm, norm_list, n=1, cutoff=threshold)
    if close:
        matched_norm = close[0]
        for col, norm in norms.items():
            if norm == matched_norm:
                logger.info("Fuzzy match via get_close_matches: %r (hint=%r)", col, hint)
                return col

    # 3. Best SequenceMatcher ratio
    best_col: Optional[str] = None
    best_ratio = 0.0
    for col, norm in norms.items():
        ratio = _fuzzy_ratio(hint, col)
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_ratio >= threshold:
        logger.info(
            "Fuzzy match via SequenceMatcher: %r (ratio=%.2f, hint=%r)",
            best_col,
            best_ratio,
            hint,
        )
        return best_col

    logger.warning(
        "No column matched hint %r (best ratio=%.2f, threshold=%.2f). "
        "Available columns: %s",
        hint,
        best_ratio,
        threshold,
        columns,
    )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_values(
    excel_path: str,
    search_value: Any,
    column_hint: Optional[str] = None,
    sheet_name: Optional[str] = None,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search for *search_value* inside an Excel file and return all matching rows.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file (.xlsx / .xlsm).
    search_value : Any
        The value to search for.  String comparison is used; the value is
        converted to str for matching.
    column_hint : str, optional
        Name (or approximate name) of the column to search in.  When provided,
        fuzzy matching is used to locate the column – the ML classifier is
        skipped entirely.  When omitted, all columns are searched.
    sheet_name : str, optional
        Sheet to read.  Defaults to the first sheet.
    fuzzy_threshold : float
        Minimum similarity ratio (0–1) for fuzzy column matching.
    case_sensitive : bool
        Whether the value search is case-sensitive.  Defaults to False.

    Returns
    -------
    list of dict
        Each dict has keys:
          - "sheet"      : name of the sheet
          - "row_index"  : 0-based position in the DataFrame
          - "excel_row"  : 1-based row number as seen in Excel (accounts for header)
          - "column"     : the column where the match was found
          - "matched_value" : the actual cell value
          - "row_data"   : the full row as a dict

    Raises
    ------
    FileNotFoundError
        If *excel_path* does not exist.
    ValueError
        If *column_hint* is provided but no matching column is found.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path!r}")

    # ── Load the sheet ────────────────────────────────────────────────────────
    read_kwargs: Dict[str, Any] = {"engine": "openpyxl"}
    if sheet_name is not None:
        read_kwargs["sheet_name"] = sheet_name

    logger.info("Reading %s …", path.name)
    df = pd.read_excel(path, **read_kwargs)

    # pd.read_excel returns a dict when sheet_name=None; normalise to a single df
    if isinstance(df, dict):
        actual_sheet = next(iter(df))
        df = df[actual_sheet]
        logger.info("Using first sheet: %r", actual_sheet)
    else:
        actual_sheet = sheet_name or "Sheet1"

    logger.info("Loaded %d rows × %d columns from sheet %r", *df.shape, actual_sheet)

    # ── Resolve the target column(s) ─────────────────────────────────────────
    columns_to_search: List[str]

    if column_hint is not None:
        matched_col = _best_column_match(
            list(df.columns.astype(str)),
            column_hint,
            threshold=fuzzy_threshold,
        )
        if matched_col is None:
            raise ValueError(
                f"Column hint {column_hint!r} did not match any column in the file. "
                f"Available columns: {list(df.columns)}"
            )
        columns_to_search = [matched_col]
        logger.info("Searching in column: %r", matched_col)
    else:
        columns_to_search = list(df.columns.astype(str))
        logger.info("No column_hint – searching all %d columns.", len(columns_to_search))

    # ── Search for the value ──────────────────────────────────────────────────
    search_str = str(search_value)
    if not case_sensitive:
        search_str = search_str.lower()

    results: List[Dict[str, Any]] = []

    for col in columns_to_search:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        if not case_sensitive:
            mask = series.str.lower() == search_str
        else:
            mask = series == search_str

        matched_rows = df[mask]
        for idx, row in matched_rows.iterrows():
            results.append(
                {
                    "sheet": actual_sheet,
                    "row_index": int(idx),
                    "excel_row": int(idx) + 2,  # +1 for header, +1 for 1-based
                    "column": col,
                    "matched_value": row[col],
                    "row_data": row.to_dict(),
                }
            )

    logger.info(
        "Search complete: %d matching row(s) found for value %r.",
        len(results),
        search_value,
    )
    return results
