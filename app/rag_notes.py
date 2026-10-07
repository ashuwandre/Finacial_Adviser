"""Simple RAG over household transaction notes.

Study note:
This is retrieval, not generation. TF-IDF ranks notes that match the question
so the writer can mention real purchases, not only yearly totals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _row_text(row: pd.Series) -> str:
    parts = [
        str(row.get("note") or ""),
        str(row.get("category") or ""),
        str(row.get("sub_category") or ""),
        str(row.get("mode") or ""),
    ]
    return " ".join(part for part in parts if part and part.lower() != "nan").strip()


def _keyword_search(df: pd.DataFrame, query: str, k: int) -> list[dict[str, Any]]:
    terms = [t for t in query.lower().split() if len(t) > 2]
    scored: list[tuple[float, dict[str, Any]]] = []
    for _, row in df.iterrows():
        text = _row_text(row).lower()
        if not text:
            continue
        score = sum(1.0 for term in terms if term in text)
        if score <= 0:
            continue
        scored.append((score, _hit(row, score)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:k]]


def _hit(row: pd.Series, score: float) -> dict[str, Any]:
    date = row.get("date")
    date_ok = date is not None and not pd.isna(date)
    year = row.get("year")
    month = row.get("month")
    return {
        "score": round(float(score), 4),
        "note": str(row.get("note") or ""),
        "category": str(row.get("category") or ""),
        "sub_category": str(row.get("sub_category") or ""),
        "amount": float(row.get("amount") or 0),
        "year": int(year) if pd.notna(year) else (int(date.year) if date_ok else None),
        "month": int(month) if pd.notna(month) else (int(date.month) if date_ok else None),
        "date": date.isoformat() if date_ok else None,
        "direction": str(row.get("direction_inferred") or row.get("direction") or ""),
    }


def search_transaction_notes(df: pd.DataFrame, query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k notes most similar to the user question."""
    if df is None or df.empty or not str(query).strip():
        return []

    working = df.copy()
    working["__doc"] = working.apply(_row_text, axis=1)
    working = working[working["__doc"].str.len() > 0]
    if working.empty:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform(working["__doc"].tolist() + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = scores.argsort()[::-1][:k]
        hits = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0:
                continue
            hits.append(_hit(working.iloc[int(index)], score))
        return hits
    except Exception:
        return _keyword_search(working, query, k)
