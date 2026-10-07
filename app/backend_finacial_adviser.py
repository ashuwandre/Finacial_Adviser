from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from data_loader import category_match_values, default_data_path, load_source
from excel_rag import phrase_matches, reload_excel_docs, search_excel_docs
from rag_notes import search_transaction_notes

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RULES_PATH = APP_DIR / "category_rules.json"

DATA_PATH_OVERRIDE: str | None = None
DF: pd.DataFrame | None = None
LOAD_META: dict[str, Any] = {}
LOAD_ERROR: str | None = None


def _jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _ok(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=_jsonable(payload), status_code=status_code)


def _load_rules() -> dict[str, str]:
    if not RULES_PATH.exists():
        return {}
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _save_rules(rules: dict[str, str]) -> None:
    RULES_PATH.write_text(json.dumps(rules, indent=4), encoding="utf-8")


def _normalize_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("transfer"):
        return "Transfer"
    if text == "income":
        return "Income"
    if text == "expense":
        return "Expense"
    return "unknown"


def _active_source() -> Path:
    if DATA_PATH_OVERRIDE:
        return Path(DATA_PATH_OVERRIDE)
    return default_data_path()


def _load_csv() -> pd.DataFrame:
    global LOAD_META
    df, LOAD_META = load_source(_active_source())
    if df.empty:
        raise ValueError(LOAD_META.get("errors") or f"No rows loaded from {_active_source()}")

    for column in ("category", "sub_category", "mode", "note", "currency", "direction"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    rules = {str(k).strip().lower(): v for k, v in _load_rules().items()}
    inferred: list[str] = []
    for _, row in df.iterrows():
        category = str(row.get("category", "")).strip().lower()
        note = str(row.get("note", "")).strip().lower()
        if category in rules:
            inferred.append(_normalize_direction(rules[category]))
        elif note in rules:
            inferred.append(_normalize_direction(rules[note]))
        elif str(row.get("direction") or "").strip():
            inferred.append(_normalize_direction(row.get("direction")))
        else:
            inferred.append("Expense")
    df["direction_inferred"] = inferred
    return df


def _get_df() -> pd.DataFrame:
    global DF, LOAD_ERROR
    if DF is None:
        try:
            DF = _load_csv()
            LOAD_ERROR = None
        except Exception as exc:
            LOAD_ERROR = str(exc)
            raise
    return DF


def _apply_filters(
    df: pd.DataFrame,
    currency: str | None,
    direction: str | None,
    category: str | None,
    mode: str | None,
    year: int | None,
    month: int | None,
    q: str | None,
) -> pd.DataFrame:
    out = df
    if currency and "currency" in out.columns:
        out = out[out["currency"].str.lower() == currency.lower()]
    if direction and "direction_inferred" in out.columns:
        out = out[out["direction_inferred"].str.lower() == direction.lower()]
    blob = pd.Series("", index=out.index)
    for column in ("category", "sub_category", "note"):
        if column in out.columns:
            blob = blob + " " + out[column].astype(str).str.lower()
    topic_masks = []
    if category:
        needles = category_match_values(category)
        topic_masks.append(blob.apply(lambda text: any(needle in text for needle in needles)))
    if q:
        phrase = str(q).lower()
        topic_masks.append(blob.apply(lambda text: phrase_matches(phrase, text)))
    if topic_masks:
        combined = topic_masks[0]
        for mask in topic_masks[1:]:
            combined = combined | mask
        out = out[combined]
    if mode and "mode" in out.columns:
        out = out[out["mode"].str.lower() == mode.lower()]
    if year is not None and "year" in out.columns:
        out = out[out["year"] == int(year)]
    if month is not None and "month" in out.columns:
        out = out[out["month"] == int(month)]
    return out


app = FastAPI(title="Financial Adviser API", version="2.0")


@app.get("/")
def home() -> JSONResponse:
    return _ok(
        {
            "service": "Financial Adviser API",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
        }
    )


@app.get("/config")
def config() -> JSONResponse:
    return _ok(
        {
            "csv_path": str(_active_source()),
            "files": LOAD_META.get("files") or [],
            "rules_path": str(RULES_PATH),
            "load_error": LOAD_ERROR,
        }
    )


@app.post("/set_csv")
async def set_csv(request: Request) -> JSONResponse:
    global DATA_PATH_OVERRIDE, DF, LOAD_ERROR
    data = await request.json()
    DATA_PATH_OVERRIDE = str((data.get("csv_path") or data.get("data_path") or "")).strip() or None
    DF = None
    LOAD_ERROR = None
    df = _get_df()
    return _ok(
        {
            "message": f"Data path set to {_active_source()}",
            "rows": int(len(df)),
            "files": LOAD_META.get("files") or [],
        }
    )


@app.get("/reload")
def reload_data() -> JSONResponse:
    global DF, LOAD_ERROR
    DF = None
    try:
        reload_excel_docs()
        _get_df()
        return _ok({"message": "Data reloaded successfully"})
    except Exception as exc:
        LOAD_ERROR = str(exc)
        return _ok({"error": f"Failed to reload data: {LOAD_ERROR}"}, status_code=500)


@app.get("/health")
def health() -> JSONResponse:
    try:
        df = _get_df()
        return _ok(
            {
                "status": "ok",
                "rows": len(df),
                "source": str(_active_source()),
                "files": LOAD_META.get("files") or [],
            }
        )
    except Exception as exc:
        return _ok({"status": "error", "load_error": str(exc)}, status_code=500)


@app.get("/info")
def info() -> JSONResponse:
    df = _get_df()
    return _ok(
        {
            "loaded": True,
            "csv_path": str(_active_source()),
            "files": LOAD_META.get("files") or [],
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "rules": _load_rules(),
            "date_min": df["date"].min() if "date" in df.columns else None,
            "date_max": df["date"].max() if "date" in df.columns else None,
            "currency_values": df["currency"].dropna().unique().tolist() if "currency" in df.columns else [],
            "direction_values": df["direction"].dropna().unique().tolist() if "direction" in df.columns else [],
        }
    )


@app.get("/filters")
def filters() -> JSONResponse:
    df = _get_df()
    years = sorted(int(y) for y in df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    months = sorted(int(m) for m in df["month"].dropna().unique().tolist()) if "month" in df.columns else []
    return _ok(
        {
            "direction_inferred_values": df["direction_inferred"].unique().tolist() if "direction_inferred" in df.columns else [],
            "sub_category_values": df["sub_category"].unique().tolist() if "sub_category" in df.columns else [],
            "currency_values": df["currency"].unique().tolist() if "currency" in df.columns else [],
            "direction_values": df["direction"].unique().tolist() if "direction" in df.columns else [],
            "category_values": df["category"].unique().tolist() if "category" in df.columns else [],
            "mode_values": df["mode"].unique().tolist() if "mode" in df.columns else [],
            "year_values": years,
            "month_values": months,
        }
    )


@app.get("/rules")
def get_rules() -> JSONResponse:
    return _ok(_load_rules())


@app.post("/rules")
async def set_rules(request: Request) -> JSONResponse:
    data = await request.json()
    rules = _load_rules()
    if "category" in data and "direction" in data:
        rules[str(data["category"])] = str(data["direction"])
    elif "categories" in data and "direction" in data:
        rules[str(data["categories"])] = str(data["direction"])
    elif isinstance(data.get("rules"), dict):
        rules.update(data["rules"])
    elif isinstance(data.get("category_rules"), dict):
        rules.update(data["category_rules"])
    _save_rules(rules)
    global DF
    DF = None
    return _ok({"message": "Rules updated successfully", "rules": rules})


@app.get("/stats")
def stats(
    direction: str | None = None,
    category: str | None = None,
    month: int | None = None,
    year: int | None = None,
    q: str | None = None,
) -> JSONResponse:
    filtered = _apply_filters(_get_df(), None, direction, category, None, year, month, q)
    by_category = (
        filtered.groupby("category", dropna=False)["amount"].sum().reset_index()
        if "category" in filtered.columns
        else pd.DataFrame(columns=["category", "amount"])
    )
    by_direction = (
        filtered.groupby("direction_inferred", dropna=False)["amount"].sum().reset_index()
        if "direction_inferred" in filtered.columns
        else pd.DataFrame(columns=["direction_inferred", "amount"])
    )

    def _sum_for(label: str) -> float:
        if by_direction.empty:
            return 0.0
        match = by_direction[by_direction["direction_inferred"].str.lower() == label]
        return float(match["amount"].sum()) if not match.empty else 0.0

    unknown_categories = []
    if "direction_inferred" in filtered.columns:
        unknown_categories = (
            filtered[filtered["direction_inferred"].str.lower() == "unknown"]["category"]
            .dropna()
            .unique()
            .tolist()
        )

    return _ok(
        {
            "by_category": by_category.to_dict(orient="records"),
            "by_direction": by_direction.to_dict(orient="records"),
            "unknown_categories": unknown_categories,
            "income_total": _sum_for("income"),
            "expense_total": _sum_for("expense"),
            "transfer_total": _sum_for("transfer"),
            "unknown_total": _sum_for("unknown"),
            "year": year,
            "month": month,
            "category": category,
            "direction": direction,
            "total_amount": float(filtered["amount"].sum()) if "amount" in filtered.columns else 0,
            "row_count": int(len(filtered)),
        }
    )


@app.get("/summary")
def summary(
    year: int | None = None,
    month: int | None = None,
    direction: str | None = None,
    currency: str | None = None,
) -> JSONResponse:
    filtered = _apply_filters(_get_df(), currency, direction, None, None, year, month, None)
    by_category = (
        filtered.groupby("category", dropna=False)["amount"].sum().reset_index()
        if "category" in filtered.columns
        else pd.DataFrame(columns=["category", "amount"])
    )
    by_direction = (
        filtered.groupby("direction_inferred", dropna=False)["amount"].sum().reset_index()
        if "direction_inferred" in filtered.columns
        else pd.DataFrame(columns=["direction_inferred", "amount"])
    )
    return _ok(
        {
            "year": year,
            "month": month,
            "direction": direction,
            "currency": currency,
            "by_category": by_category.to_dict(orient="records"),
            "by_direction": by_direction.to_dict(orient="records"),
            "total_amount": float(filtered["amount"].sum()) if "amount" in filtered.columns else 0,
            "row_count": int(len(filtered)),
        }
    )


@app.get("/transactions")
def transactions(
    currency: str | None = None,
    direction: str | None = None,
    category: str | None = None,
    mode: str | None = None,
    year: int | None = None,
    month: int | None = None,
    q: str | None = None,
    limit: int | None = 50,
) -> JSONResponse:
    filtered = _apply_filters(_get_df(), currency, direction, category, mode, year, month, q)
    if limit is not None:
        filtered = filtered.head(int(limit))
    records = filtered.to_dict(orient="records")
    return _ok({"transactions": records, "row_count": len(records)})


@app.get("/search_notes")
def search_notes(
    q: str,
    k: int = 12,
    year: int | None = None,
    category: str | None = None,
) -> JSONResponse:
    """RAG over the three Excel workbooks, with a pandas note search fallback."""
    hits = search_excel_docs(q, k=max(1, min(int(k), 20)), year=year)
    if not hits:
        filtered = _apply_filters(_get_df(), None, None, category, None, year, None, q)
        hits = search_transaction_notes(filtered, q, k=max(1, min(int(k), 20)))
    return _ok({"query": q, "hits": hits, "hit_count": len(hits), "source": "excel_rag"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_finacial_adviser:app", host="127.0.0.1", port=8000, reload=False)
