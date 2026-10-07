"""RAG over the three household Excel workbooks.

Each sheet is turned into text chunks. A question retrieves the closest
chunks (TF-IDF). The LLM then answers only from those chunks.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPENSES_DIR = PROJECT_ROOT / "My Expenses"

ALLOWED_FILES = (
    "6_Yearly Expenses Tracker.xlsx",
    "7_Total Expenditure.xlsx",
    "8_Krisala 41 Cosmo A-601.xlsx",
)

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return text


def _row_chunks_yearly(df: pd.DataFrame, file_name: str, sheet: str) -> list[dict[str, Any]]:
    header_i = None
    for index in range(min(6, len(df))):
        values = [_cell(v).lower() for v in df.iloc[index].tolist()]
        if sum(1 for v in values if v[:3] in MONTHS) >= 6:
            header_i = index
            break
    if header_i is None:
        return []
    header = [_cell(v) for v in df.iloc[header_i].tolist()]
    last_category = ""
    docs: list[dict[str, Any]] = []
    for _, raw in df.iloc[header_i + 1 :].iterrows():
        values = [_cell(v) for v in raw.tolist()]
        if not any(values):
            continue
        paired = list(zip(header, values))
        category = ""
        item = ""
        months: list[str] = []
        total = ""
        year_cols: list[str] = []
        for title, value in paired:
            key = title.lower()
            if not value:
                if "category" in key and "total" not in key:
                    pass
                continue
            if "category" in key and "total" not in key:
                category = value
                last_category = value
            elif "type" in key:
                item = value
            elif key[:3] in MONTHS:
                months.append(f"{title}={value}")
            elif key.isdigit() and len(key) == 4:
                year_cols.append(f"{title}={value}")
            elif "total" in key:
                total = value
        if not item:
            item = next((v for v in values if v and v.lower() not in {"total"}), "")
        if not item:
            continue
        category = category or last_category
        parts = [
            f"File: {file_name}",
            f"Sheet: {sheet}",
            f"Year: {sheet}" if re.fullmatch(r"20\d{2}", sheet) else "",
            f"Expense category: {category}" if category else "",
            f"Expense type: {item}",
        ]
        if months:
            parts.append("Monthly amounts: " + ", ".join(months))
        if year_cols:
            parts.append("Year amounts: " + ", ".join(year_cols))
        if total:
            parts.append(f"Total: {total}")
        text = " | ".join(part for part in parts if part)
        amount = 0.0
        if total:
            try:
                amount = abs(float(str(total).replace(",", "")))
            except ValueError:
                amount = 0.0
        docs.append(
            {
                "text": text,
                "file": file_name,
                "sheet": sheet,
                "note": item,
                "category": category,
                "amount": amount,
                "year": int(sheet) if re.fullmatch(r"20\d{2}", sheet) else None,
            }
        )
    return docs


def _row_chunks_generic(df: pd.DataFrame, file_name: str, sheet: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    header = [_cell(v) for v in df.iloc[0].tolist()] if len(df) else []
    start = 1 if any(header) else 0
    for ridx, raw in df.iloc[start:].iterrows():
        values = [_cell(v) for v in raw.tolist()]
        pairs = []
        for idx, value in enumerate(values):
            if not value:
                continue
            title = header[idx] if idx < len(header) and header[idx] else f"col{idx}"
            pairs.append(f"{title}={value}")
        if len(pairs) < 2:
            continue
        text = f"File: {file_name} | Sheet: {sheet} | " + " | ".join(pairs)
        amount = 0.0
        for value in values:
            try:
                number = float(value.replace(",", ""))
                if number > amount:
                    amount = number
            except ValueError:
                continue
        docs.append(
            {
                "text": text,
                "file": file_name,
                "sheet": sheet,
                "note": " | ".join(pairs[:4]),
                "category": sheet,
                "amount": amount,
                "year": None,
            }
        )
    return docs


def _sheet_preview(df: pd.DataFrame, file_name: str, sheet: str) -> dict[str, Any]:
    preview = df.fillna("").astype(str).head(25)
    lines = ["\t".join(_cell(v) for v in row) for row in preview.values.tolist()]
    text = f"File: {file_name} | Sheet: {sheet} | Table preview:\n" + "\n".join(lines)
    return {
        "text": text,
        "file": file_name,
        "sheet": sheet,
        "note": f"{sheet} preview",
        "category": sheet,
        "amount": 0,
        "year": int(sheet) if re.fullmatch(r"20\d{2}", sheet) else None,
    }


def _chunk_file(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        if df.empty:
            continue
        docs.append(_sheet_preview(df, path.name, sheet))
        yearly = _row_chunks_yearly(df, path.name, sheet)
        docs.extend(yearly if yearly else _row_chunks_generic(df, path.name, sheet))
    return docs


@lru_cache(maxsize=1)
def load_excel_docs() -> tuple[dict[str, Any], ...]:
    docs: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in ALLOWED_FILES:
        path = EXPENSES_DIR / name
        if not path.exists():
            errors.append(f"missing {name}")
            continue
        try:
            docs.extend(_chunk_file(path))
        except PermissionError:
            errors.append(f"{name} is open in Excel; close it and reload")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        docs.append(
            {
                "text": "Load notes: " + "; ".join(errors),
                "file": "",
                "sheet": "",
                "note": "loader",
                "category": "",
                "amount": 0,
                "year": None,
            }
        )
    return tuple(docs)


_tfidf_cache: dict[str, Any] = {}


def reload_excel_docs() -> None:
    load_excel_docs.cache_clear()
    _tfidf_cache.clear()
    load_excel_docs()


WEAK_STOP = {
    "a", "an", "and", "also", "buy", "can", "check", "did", "do", "does", "expense",
    "expenses", "find", "for", "how", "i", "in", "much", "my", "of", "on", "please",
    "send", "sent", "show", "spend", "spent", "the", "to", "today", "total", "we",
    "what", "whether", "year",
}


def normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _stem(token: str) -> str:
    word = re.sub(r"[^a-z0-9]", "", (token or "").casefold())
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        core = word[:-3]
        if len(core) >= 2 and core[-1] == core[-2]:
            return core[:-1]
        return core
    if word.endswith("ed") and len(word) > 4:
        core = word[:-2]
        if len(core) >= 2 and core[-1] == core[-2]:
            return core[:-1]
        return core
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _tokens_similar(left: str, right: str) -> bool:
    a = (left or "").casefold()
    b = (right or "").casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and (shorter in longer or longer.startswith(shorter)):
        return True
    shared = 0
    for index in range(min(len(a), len(b))):
        if a[index] != b[index]:
            break
        shared += 1
    extra = longer[shared:]
    if len(shorter) >= 4 and shared >= 3 and extra in {"s", "es", "ed", "ing", "ings", "ies", "er", "ers"}:
        return True
    if shorter.endswith("e") and shorter[:-1] + "ing" == longer:
        return True
    if shorter + "ing" == longer or shorter + "ed" == longer:
        return True
    sa, sb = _stem(a), _stem(b)
    if sa == sb:
        return True
    if min(len(sa), len(sb)) >= 4 and (sa.startswith(sb) or sb.startswith(sa)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.84 and min(len(a), len(b)) >= 4


def phrase_matches(query: str, text: str) -> bool:
    """True when query words match text even if forms differ (dine vs dining)."""
    q = normalize(query)
    blob = normalize(text)
    if not q or not blob:
        return False
    if q in blob or blob in q:
        return True
    wanted = [tok for tok in re.findall(r"[a-z0-9]+", q) if tok not in WEAK_STOP and len(tok) > 1]
    have = re.findall(r"[a-z0-9]+", blob)
    if not wanted:
        return False
    return all(any(_tokens_similar(token, other) for other in have) for token in wanted)


def _is_preview(doc: dict[str, Any]) -> bool:
    note = normalize(str(doc.get("note") or ""))
    return note.endswith("preview") or "table preview" in normalize(str(doc.get("text") or ""))


def _lexical_score(query: str, doc: dict[str, Any], text: str) -> float:
    note = normalize(str(doc.get("note") or ""))
    category = normalize(str(doc.get("category") or ""))
    score = 0.0
    if phrase_matches(query, note):
        score += 2.5
    elif phrase_matches(query, f"{note} {category}"):
        score += 1.6
    elif phrase_matches(query, text):
        score += 0.8
    note_ratio = SequenceMatcher(None, query, note).ratio() if note else 0.0
    if note_ratio >= 0.72:
        score += note_ratio
    return score


def _cosine_scores(texts: list[str], query: str) -> list[float] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    key = f"{len(texts)}:{texts[0][:60] if texts else ''}:{texts[-1][:60] if texts else ''}"
    packed = _tfidf_cache.get(key)
    try:
        if packed is None:
            word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True)
            char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
            packed = (word_vec, word_vec.fit_transform(texts), char_vec, char_vec.fit_transform(texts))
            _tfidf_cache.clear()
            _tfidf_cache[key] = packed
        word_vec, word_mat, char_vec, char_mat = packed
        word_scores = cosine_similarity(word_vec.transform([query]), word_mat).ravel()
        char_scores = cosine_similarity(char_vec.transform([query]), char_mat).ravel()
    except Exception:
        return None
    return [0.5 * float(word) + 0.8 * float(char) for word, char in zip(word_scores, char_scores)]


def search_excel_docs(query: str, k: int = 12, year: int | None = None) -> list[dict[str, Any]]:
    docs = list(load_excel_docs())
    if not docs or not str(query).strip():
        return []
    texts = [normalize(str(doc.get("text") or "")) for doc in docs]
    q = normalize(str(query))
    scores = [_lexical_score(q, doc, text) for doc, text in zip(docs, texts)]
    cosine = _cosine_scores(texts, q)
    if cosine:
        scores = [lex + sim for lex, sim in zip(scores, cosine)]
    ranked = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    hits: list[dict[str, Any]] = []
    year_text = str(year) if year else ""
    for index in ranked:
        doc = dict(docs[index])
        score = float(scores[index])
        text = texts[index]
        if _is_preview(doc):
            score *= 0.2
        if year_text and year_text in text:
            score += 0.2
        if score <= 0:
            continue
        if year and doc.get("year") not in {None, int(year)} and year_text not in text:
            continue
        doc["score"] = round(score, 4)
        hits.append(doc)
        if len(hits) >= k:
            break
    return hits


def merge_hits(hit_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for hits in hit_lists:
        for hit in hits:
            if _is_preview(hit):
                continue
            key = (
                str(hit.get("file") or ""),
                str(hit.get("sheet") or ""),
                normalize(str(hit.get("note") or hit.get("text") or "")),
            )
            current = merged.get(key)
            if current is None or float(hit.get("score") or 0) > float(current.get("score") or 0):
                merged[key] = hit
    return sorted(merged.values(), key=lambda item: float(item.get("score") or 0), reverse=True)


def line_items_for_lookups(hits: list[dict[str, Any]], lookups: list[str]) -> list[dict[str, Any]]:
    phrases = [normalize(item) for item in lookups if normalize(item)]
    if not phrases:
        return [hit for hit in hits if float(hit.get("amount") or 0) > 0 and not _is_preview(hit)]
    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        if _is_preview(hit) or float(hit.get("amount") or 0) <= 0:
            continue
        blob = " ".join(str(hit.get(key) or "") for key in ("note", "category", "text"))
        if not any(phrase_matches(phrase, blob) for phrase in phrases):
            continue
        key = (str(hit.get("file") or ""), str(hit.get("sheet") or ""), normalize(str(hit.get("note") or "")))
        if key in seen:
            continue
        seen.add(key)
        picked.append(hit)
    yearly = [hit for hit in picked if hit.get("year")]
    return yearly or picked


def sum_amounts(hits: list[dict[str, Any]]) -> float:
    return round(sum(float(hit.get("amount") or 0) for hit in hits), 2)
