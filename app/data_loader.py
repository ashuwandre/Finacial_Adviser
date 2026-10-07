"""Load any household ledger: CSV, Excel, or a folder of files.

Normalizes rows to:
date, year, month, amount, category, sub_category, note, mode, currency, direction
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MY_EXPENSES_DIR = PROJECT_ROOT / "My Expenses"
DEFAULT_CSV_PATH = APP_DIR / "default_data.csv"
FALLBACK_CSV_PATH = PROJECT_ROOT / "pandas" / "Daily Household Transactions.csv"

ALLOWED_FILE_HINTS = (
    "yearly expenses tracker",
    "total expenditure",
    "krisala 41 cosmo",
)
SKIP_SHEET_HINTS = ()

COLUMN_ALIASES = {
    "date": "date",
    "txn date": "date",
    "transaction date": "date",
    "value date": "date",
    "amount": "amount",
    "amt": "amount",
    "debit": "amount",
    "inr": "amount",
    "rs": "amount",
    "price": "amount",
    "currency": "currency",
    "direction": "direction",
    "income/expense": "direction",
    "type": "direction",
    "dr/cr": "direction",
    "category": "category",
    "expense category": "category",
    "sub-category": "sub_category",
    "subcategory": "sub_category",
    "sub_category": "sub_category",
    "expense type": "sub_category",
    "mode": "mode",
    "payment mode": "mode",
    "account": "mode",
    "note": "note",
    "notes": "note",
    "narration": "note",
    "description": "note",
    "particulars": "note",
    "details": "note",
    "item": "note",
}

MONTH_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

CATEGORY_KEYWORDS = (
    (("family support", "friend/family", "pappa", "papa", "mummy", "aai", "nana", "mama", "kaka", "family"), "Family"),
    (("dmart", "grocery", "groceries", "fruit", "vegetable", "snack", "food"), "Food"),
    (("school", "education", "donation", "dance", "badminton", "gym", "cricket", "fee"), "Education"),
    (("rent", "housing"), "Housing"),
    (("emi", "loan"), "Loan"),
    (("lic", "sip", "insurance", "tata aia", "tata term"), "Insurance"),
    (("petrol", "fuel", "train", "ticket", "uber"), "Transportation"),
    (("trip", "tourism", "holiday"), "Tourism"),
    (("electric", "light bill", "internet", "wifi", "wi-fi", "phone", "mobile"), "Utilities"),
    (("gas",), "Utilities"),
    (("maid", "house help", "cook"), "Household"),
    (("dental", "hospital", "medicine", "treatment", "injection", "health"), "Health"),
    (("credit card",), "Credit Card"),
    (("krisala",), "Property"),
)

SKIP_ROW_LABELS = {
    "total",
    "grand total",
    "total yearly",
    "overall",
    "overall expenses",
    "expense category wise yearly total",
    "month payment",
}

DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def default_data_path() -> Path:
    env = (os.getenv("DATA_PATH") or "").strip().strip('"')
    if env:
        return Path(env)
    if MY_EXPENSES_DIR.exists():
        return MY_EXPENSES_DIR
    if DEFAULT_CSV_PATH.exists():
        return DEFAULT_CSV_PATH
    return FALLBACK_CSV_PATH


def _norm_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def infer_category(text: str, fallback: str = "Other") -> str:
    lowered = (text or "").lower()
    for keys, label in CATEGORY_KEYWORDS:
        if any(key in lowered for key in keys):
            return label
    cleaned = str(text or "").strip()
    return cleaned.title() if cleaned else fallback


def category_match_values(category: str) -> list[str]:
    key = (category or "").strip().lower()
    values = {key}
    for keys, label in CATEGORY_KEYWORDS:
        if key == label.lower() or key in keys:
            values.update(keys)
            values.add(label.lower())
    return [item for item in values if item]


def _skip_file(path: Path) -> bool:
    name = path.stem.lower()
    return not any(hint in name for hint in ALLOWED_FILE_HINTS)


def _skip_sheet(name: str) -> bool:
    lowered = (name or "").lower()
    return any(hint in lowered for hint in SKIP_SHEET_HINTS)


def _year_from_filename(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.stem)
    return int(match.group(1)) if match else None


def parse_sheet_period(sheet: str, fallback_year: int | None = None) -> tuple[int | None, int | None]:
    text = str(sheet or "").strip()
    if re.fullmatch(r"20\d{2}", text):
        return int(text), None
    match = re.search(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|jul(?:y)?|"
        r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[a-z]*[-_\s]*(\d{2,4})",
        text,
        flags=re.I,
    )
    if not match:
        if fallback_year and text.isdigit() and len(text) == 4:
            return int(text), None
        return fallback_year, None
    token = match.group(1).lower()
    month = MONTH_NUM.get(token) or MONTH_NUM.get(token[:4]) or MONTH_NUM.get(token[:3])
    year = int(match.group(2))
    if year < 100:
        year += 2000
    return year, month


def _is_numeric_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.notna().sum()) >= max(2, int(len(series) * 0.3))


def _clean_amount(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "")
    if not text or text.lower() in SKIP_ROW_LABELS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _row_label(value: Any) -> str:
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
    return text


def _should_skip_label(label: str) -> bool:
    text = _norm_header(label)
    if not text:
        return True
    if text in SKIP_ROW_LABELS:
        return True
    return "month payment" in text or text.startswith("overall")


def _make_row(
    *,
    amount: float,
    note: str,
    category: str,
    sub_category: str = "",
    year: int | None,
    month: int | None,
    direction: str = "Expense",
    source_file: str = "",
    source_sheet: str = "",
) -> dict[str, Any]:
    date = None
    if year and month:
        date = pd.Timestamp(year=int(year), month=int(month), day=1)
    return {
        "date": date,
        "year": int(year) if year else None,
        "month": int(month) if month else None,
        "amount": abs(float(amount)),
        "category": category or "Other",
        "sub_category": sub_category or note,
        "note": note,
        "mode": "",
        "currency": "INR",
        "direction": direction,
        "source_file": source_file,
        "source_sheet": source_sheet,
    }


def _parse_standard(df: pd.DataFrame, source_file: str, source_sheet: str) -> list[dict[str, Any]]:
    renamed = {}
    for column in df.columns:
        key = _norm_header(column)
        if key in COLUMN_ALIASES:
            renamed[column] = COLUMN_ALIASES[key]
    work = df.rename(columns=renamed).copy()
    work.columns = [str(c).strip() for c in work.columns]
    if "amount" not in work.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, raw in work.iterrows():
        amount = _clean_amount(raw.get("amount"))
        if amount is None or amount == 0:
            continue
        note = _row_label(raw.get("note") or raw.get("sub_category") or raw.get("category"))
        if _should_skip_label(note) and not _row_label(raw.get("category")):
            continue
        category = _row_label(raw.get("category")) or infer_category(note)
        sub = _row_label(raw.get("sub_category"))
        direction = _row_label(raw.get("direction")) or "Expense"
        date = raw.get("date")
        year = month = None
        parsed_date = pd.to_datetime(date, dayfirst=True, errors="coerce") if date is not None else pd.NaT
        if pd.notna(parsed_date):
            year, month = int(parsed_date.year), int(parsed_date.month)
        rows.append(
            _make_row(
                amount=amount,
                note=note or category,
                category=category,
                sub_category=sub,
                year=year,
                month=month,
                direction=direction or "Expense",
                source_file=source_file,
                source_sheet=source_sheet,
            )
        )
        if pd.notna(parsed_date):
            rows[-1]["date"] = parsed_date
    return rows


def _find_header_row(df: pd.DataFrame) -> int | None:
    for index in range(min(8, len(df))):
        values = [_norm_header(v) for v in df.iloc[index].tolist()]
        joined = " ".join(values)
        if "expense category" in joined and "expense type" in joined:
            return index
        if "category" in values and ("amount" in values or "expense type" in values):
            return index
    return None


def _parse_category_type_table(
    df: pd.DataFrame,
    year: int | None,
    month: int | None,
    source_file: str,
    source_sheet: str,
    header_row: int,
) -> list[dict[str, Any]]:
    header = [_norm_header(v) for v in df.iloc[header_row].tolist()]
    body = df.iloc[header_row + 1 :].copy()
    cat_col = next((i for i, name in enumerate(header) if name in {"expense category", "category"}), 0)
    type_col = next((i for i, name in enumerate(header) if name in {"expense type", "sub_category", "note"}), 1)
    amount_col = None
    for index, name in enumerate(header):
        if index in {cat_col, type_col}:
            continue
        if name in MONTH_NUM or name in {"amount", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept"}:
            amount_col = index
            break
    if amount_col is None:
        amount_col = min(type_col + 1, body.shape[1] - 1)

    last_category = ""
    rows: list[dict[str, Any]] = []
    for _, raw in body.iterrows():
        values = list(raw.values)
        if cat_col < len(values) and _row_label(values[cat_col]):
            last_category = _row_label(values[cat_col])
        label = _row_label(values[type_col]) if type_col < len(values) else ""
        if _should_skip_label(label) or _should_skip_label(last_category):
            continue
        amount = _clean_amount(values[amount_col] if amount_col < len(values) else None)
        if amount is None or amount == 0:
            continue
        category = last_category or infer_category(label)
        if category.lower() in {"nan", "overall expenses"}:
            category = infer_category(label)
        rows.append(
            _make_row(
                amount=amount,
                note=label,
                category=category,
                sub_category=label,
                year=year,
                month=month,
                source_file=source_file,
                source_sheet=source_sheet,
            )
        )
    return rows


def _parse_simple_list(
    df: pd.DataFrame,
    year: int | None,
    month: int | None,
    source_file: str,
    source_sheet: str,
) -> list[dict[str, Any]]:
    work = df.dropna(how="all", axis=1).dropna(how="all")
    if work.empty:
        return []
    text_col = amount_col = None
    for column in work.columns:
        series = work[column]
        if amount_col is None and _is_numeric_series(series):
            amount_col = column
        elif text_col is None and series.astype(str).str.len().mean() >= 1:
            text_col = column
    if amount_col is None:
        return []
    if text_col is None:
        remaining = [c for c in work.columns if c != amount_col]
        text_col = remaining[0] if remaining else amount_col

    rows: list[dict[str, Any]] = []
    for _, raw in work.iterrows():
        label = _row_label(raw.get(text_col))
        if _should_skip_label(label):
            continue
        amount = _clean_amount(raw.get(amount_col))
        if amount is None or amount == 0:
            continue
        rows.append(
            _make_row(
                amount=amount,
                note=label,
                category=infer_category(label),
                sub_category=label,
                year=year,
                month=month,
                source_file=source_file,
                source_sheet=source_sheet,
            )
        )
    return rows


def _yearly_header_row(df: pd.DataFrame) -> int | None:
    for index in range(min(6, len(df))):
        values = [_norm_header(v) for v in df.iloc[index].tolist()]
        month_hits = sum(1 for value in values if value in MONTH_NUM)
        if month_hits >= 6:
            return index
    return None


def _parse_yearly_matrix(
    df: pd.DataFrame,
    year: int | None,
    source_file: str,
    source_sheet: str,
) -> list[dict[str, Any]]:
    header_row = _yearly_header_row(df)
    if header_row is None or year is None:
        return []
    header = [_norm_header(v) for v in df.iloc[header_row].tolist()]
    cat_col = next((i for i, name in enumerate(header) if "category" in name and "total" not in name), None)
    type_col = next((i for i, name in enumerate(header) if "type" in name or name in {"note", "item"}), None)
    month_cols = [(i, MONTH_NUM[name]) for i, name in enumerate(header) if name in MONTH_NUM]
    total_col = next((i for i, name in enumerate(header) if name in {"total yearly", "total"}), None)
    if type_col is None or not month_cols:
        return []
    last_category = ""
    rows: list[dict[str, Any]] = []
    for _, raw in df.iloc[header_row + 1 :].iterrows():
        values = list(raw.values)
        if cat_col is not None and cat_col < len(values) and _row_label(values[cat_col]):
            last_category = _row_label(values[cat_col])
        label = _row_label(values[type_col]) if type_col < len(values) else ""
        if _should_skip_label(label) or _should_skip_label(last_category):
            continue
        monthly = []
        for index, month in month_cols:
            amount = _clean_amount(values[index] if index < len(values) else None)
            if amount:
                monthly.append((month, amount))
        if monthly:
            for month, amount in monthly:
                rows.append(
                    _make_row(
                        amount=amount,
                        note=label,
                        category=last_category or infer_category(label),
                        sub_category=label,
                        year=year,
                        month=month,
                        source_file=source_file,
                        source_sheet=source_sheet,
                    )
                )
            continue
        if total_col is not None:
            total = _clean_amount(values[total_col] if total_col < len(values) else None)
            if total:
                rows.append(
                    _make_row(
                        amount=total,
                        note=label,
                        category=last_category or infer_category(label),
                        sub_category=label,
                        year=year,
                        month=None,
                        source_file=source_file,
                        source_sheet=source_sheet,
                    )
                )
    return rows


def _parse_sheet(df: pd.DataFrame, path: Path, sheet: str) -> list[dict[str, Any]]:
    year, month = parse_sheet_period(sheet, _year_from_filename(path))
    yearly_rows = _parse_yearly_matrix(df, year, path.name, sheet)
    if yearly_rows:
        return yearly_rows
    header_row = _find_header_row(df)
    if header_row is not None:
        mapped = {i: df.iloc[header_row, i] for i in range(df.shape[1])}
        titled = df.iloc[header_row + 1 :].copy()
        titled.columns = [mapped.get(i, i) for i in range(df.shape[1])]
        standard = _parse_standard(titled, path.name, sheet)
        if standard:
            for row in standard:
                row["year"] = row["year"] or year
                row["month"] = row["month"] or month
                if row["date"] is None and row["year"] and row["month"]:
                    row["date"] = pd.Timestamp(year=int(row["year"]), month=int(row["month"]), day=1)
            return standard
        return _parse_category_type_table(df, year, month, path.name, sheet, header_row)
    standard = _parse_standard(df, path.name, sheet)
    if standard:
        return standard
    return _parse_simple_list(df, year, month, path.name, sheet)


def _read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, header=None)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        rows = _parse_standard(df, path.name, "")
        if rows:
            return rows
        return _parse_sheet(df, path, path.stem)
    if suffix not in {".xlsx", ".xls"}:
        return []
    try:
        xl = pd.ExcelFile(path)
    except PermissionError as exc:
        raise PermissionError(f"{path.name} is open in Excel. Close it and reload.") from exc
    rows: list[dict[str, Any]] = []
    for sheet in xl.sheet_names:
        if _skip_sheet(sheet):
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        rows.extend(_parse_sheet(df, path, sheet))
    return rows


def iter_data_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in DATA_EXTENSIONS and not _skip_file(path):
            yield path


def load_source(source: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(source) if source is not None else default_data_path()
    files = list(iter_data_files(path)) if path.is_dir() else ([path] if path.exists() else [])
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for file_path in files:
        try:
            records.extend(load_file(file_path))
        except Exception as exc:
            errors.append(f"{file_path.name}: {exc}")

    frame = pd.DataFrame(records)
    if frame.empty:
        meta = {
            "source": str(path),
            "files": [str(item) for item in files],
            "rows": 0,
            "errors": errors or [f"No usable rows in {path}"],
        }
        return frame, meta

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["year"] = frame["year"].fillna(frame["date"].dt.year)
        frame["month"] = frame["month"].fillna(frame["date"].dt.month)
    for column in ("category", "sub_category", "note", "mode", "currency", "direction", "source_file", "source_sheet"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    meta = {
        "source": str(path),
        "files": [file_path.name for file_path in files],
        "rows": int(len(frame)),
        "errors": errors,
        "years": sorted(int(y) for y in frame["year"].dropna().unique()),
    }
    return frame, meta
