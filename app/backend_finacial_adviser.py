from __future__ import annotations
from locale import currency
from calendar import month
from typing import Any
import json
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

APP_ROOT = Path(__file__).resolve().parent[1]
RULES_PATH = APP_ROOT / "app" / "category_rules.json"
DEFAULT_CSV_PATH = (APP_ROOT / "app" / "default_data.csv")

CSV_PATH_OVERRIDE : str| None = None

def _load_rules():
    if not RULES_PATH.exists():
        return {}
    text = RULES_PATH.read_text()
    return json.loads(text)

def _save_rules(rules):
    RULES_PATH.write_text(json.dumps(rules, indent=4))
  

        
def _load_csv() -> pd.DataFrame:
    csv_path = CSV_PATH_OVERRIDE if CSV_PATH_OVERRIDE else DEFAULT_CSV_PATH
    csv_path = Path(csv_path)
    
    df = pd.read_csv(csv_path)
    df =df.rename(columns = {'Date': 'date', 'Amount': 'amount', 'Currency': 'currency', 'Direction': 'direction', 'Category': 'category', 'Sub-Category': 'sub_category', 'Mode': 'mode', 'Note': 'note'})
    df['date']= pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['amount']= pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    for c in df.columns:
        if c not in df.columns:
            df[c] = df[c].fillna("").astype(str)
            
    rules = _load_rules()
    if "category" in df.columns:
        df["direction_inferred"] = df["category"].map(rules).fillna('unknown').astype(str)
        
    
    return df

DF =pd.DataFrame |None = None
LOAD_ERROR : str| None = None

def _get_df() -> pd.DataFrame:
    global DF, LOAD_ERROR
    if DF is None:
        try:
            DF = _load_csv()
        except Exception as e:
            LOAD_ERROR = str(e)
            raise
    return DF

def _apply_filters(df,currency, direction, category, mode, year, month, q):

            
        out = df
        
        if currency and "currency" in df.columns:
            out = out[df["currency"] == currency]
            
        if direction and 'directions' in df.columns:
            out = out[df["direction"] == direction]
        if category and "category" in df.columns:
            out = out[df["category"] == category]
        if mode and "mode" in df.columns:
            out = out[df["mode"] == mode]
        if year is not None and "year" in df.columns:
            out = out[df["year"] == year]
        if month is not None and "month" in df.columns:
            out = out[df["month"] == month]
            
        if q:
            ql=q.lower()
            out = out[out.apply(lambda row: row.astype(str).str.lower().str.contains(ql).any(), axis=1)] 
                    
        return out

app = FastAPI(title="Financial Adviser API", version="1.0")

@app.get("/config")
def config() -> dict[str, Any]:
    current = CSV_PATH_OVERRIDE if CSV_PATH_OVERRIDE else str(DEFAULT_CSV_PATH)
    return {
        "csv_path": current,
        "rules_path": str(RULES_PATH),
        "load_error": LOAD_ERROR,
    }
    
@app.post('/set_csv')
async def set_csv(request: Request) -> JSONResponse:
    global CSV_PATH_OVERRIDE, DF
    data = await request.json()
    csv_path_override = str((data.get("csv_path") or "").strip())
    DF = None
    df = _get_df()
    return {"message": f"CSV path set to {csv_path_override}"}
   

@app.get('/reload')
def reload() -> JSONResponse:
    global DF, LOAD_ERROR
    try:
        DF = _get_df()
        LOAD_ERROR = None
        return JSONResponse(content={"message": "Data reloaded successfully"}, status_code=200)
    except Exception as e:
        LOAD_ERROR = str(e)
        return JSONResponse(content={"error": f"Failed to reload data: {LOAD_ERROR}"}, status_code=500)
    
    
@app.get('/health')
def health() -> JSONResponse:
    if LOAD_ERROR:
        return JSONResponse(content={"status": "error", "load_error": LOAD_ERROR}, status_code=500)
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get('/info')
def info() -> JSONResponse:
    
    df = _get_df()
    date_min = df['date'].min() if 'date' in df.columns else None
    date_max = df['date'].max() if 'date' in df.columns else None
    info = {
        'loaded': True,
        'csv_path': os.getenv("CSV_PATH_OVERRIDE", str(DEFAULT_CSV_PATH)),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "rules": _load_rules(),
        "date_min": date_min,
        "date_max": date_max,
        'currency_values': df['currency'].unique().tolist() if 'currency' in df.columns else [],
        'direction_values': df['direction'].unique().tolist() if 'direction' in df.columns else []
    }
    return JSONResponse(content=info, status_code=200)

@app.get('/filters')
def filters()->JSONResponse:
    df = _get_df()
    filters = {
        'direction_inferred_values': df['direction_inferred'].unique().tolist() if 'direction_inferred' in df.columns else [],
        'sub_category_values': df['sub_category'].unique().tolist() if 'sub_category' in df.columns else [],
        'currency_values': df['currency'].unique().tolist() if 'currency' in df.columns else [],
        'direction_values': df['direction'].unique().tolist() if 'direction' in df.columns else [],
        'category_values': df['category'].unique().tolist() if 'category' in df.columns else [],
        'mode_values': df['mode'].unique().tolist() if 'mode' in df.columns else [],
        'year_values': sorted(df['year'].dropna().unique().tolist()) if 'year' in df.columns else [],
        'month_values': sorted(df['month'].dropna().unique().tolist()) if 'month' in df.columns else []
    }
    return JSONResponse(content=filters, status_code=200)

@app.get('/rules')
async def get_rules() -> JSONResponse:
    rules = _load_rules()
    return JSONResponse(content=rules, status_code=200)

@app.post('/rules')
async def set_rules(request: Request) -> JSONResponse:
    data = await request.json()
    category_rules = data.get("category_rules", {})
    direction_rules = data.get("direction_rules", {})
    rules = _load_rules()
    rules[category_rules] = direction_rules
    _save_rules(rules)
    return JSONResponse(content={"message": "Rules updated successfully"}, status_code=200)

@app.get('/stats')
def stats( direction: str | None = None, category: str | None = None, month: int | None = None, year: int | None = None) -> JSONResponse:
    base = _get_df()
    filtered = _apply_filters(base, None, direction, category, None, year, month, None)
    if direction:
        df =df[df['direction_inferred'].str.lower() == direction.lower()]
    by_category = filtered.groupby('category')['amount'].sum().reset_index()
    dd = df.groupby('direction_inferred')['amount'].sum().reset_index()
    income = dd[dd['direction_inferred'].str.lower() == 'income']['amount'].sum() if 'income' in dd['direction_inferred'].str.lower().values else 0
    expense = dd[dd['direction_inferred'].str.lower() == 'expense']['amount'].sum() if 'expense' in dd['direction_inferred'].str.lower().values else 0
    transfer = dd[dd['direction_inferred'].str.lower() == 'transfer']['amount'].sum() if 'transfer' in dd['direction_inferred'].str.lower().values else 0
    unknown = dd[dd['direction_inferred'].str.lower() == 'unknown']['amount'].sum() if 'unknown' in dd['direction_inferred'].str.lower().values else 0
    
    unknown_categories = filtered[filtered['direction_inferred'].str.lower() == 'unknown']['category'].unique().tolist() if 'unknown' in filtered['direction_inferred'].str.lower().values else []
    return JSONResponse(content={
        "by_category": by_category.to_dict(orient='records'),
        "by_direction": dd.to_dict(orient='records'),
        "unknown_categories": unknown_categories,
        'income_total': income,
        'expense_total': expense,
        'transfer_total': transfer,
        'unknown_total': unknown
        'year': year,
        'month': month
        'total_amount': filtered['amount'].sum() if 'amount' in filtered.columns else 0
        
    }, status_code=200)
    
    
@app.get('/summary')
def summary(
    year = None
    month = None
    direction = None
    currency = None):
    
    base = _get_df()
    filtered = _apply_filters(base, currency, direction, None, None, year, month, q=None)
    by_category = filtered.groupby('category')['amount'].sum().reset_index()
    
    return JSONResponse(content={"year": year, "month": month, "by_category": by_category.to_dict(orient='records'),
                                 "by_direction": by_direction.to_dict(orient='records'),
                                 "currency": currency,"total_amount": filtered['amount'].sum() if 'amount' in filtered.columns else 0}, status_code=200)
    
    
@app.get('/transactions')
def transactions(
    currency: str | None = None,
    direction: str | None = None,
    category: str | None = None,
    mode: str | None = None,
    year: int | None = None,
    month: int | None = None,
    q: str | None = None,
    limit: int | None = None,
) -> JSONResponse:
    base = _get_df()
    filtered = _apply_filters(base, currency, direction, category, mode, year, month, q)
    
    if limit is not None:
        filtered = filtered.head(limit)
    
    return JSONResponse(content={"transactions": filtered.to_dict(orient='records')}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_finacial_adviser:app", host="127.0.0.1", port=8000, reload=True)