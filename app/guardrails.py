"""Study guardrails for the finance team.

Business rule (simple):
1. Never send email unless the user gave an address.
2. Never invent income/expense amounts. Use tool or RAG numbers only.
3. Final numbers must fit {income, expense, source_tool}.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
EMAIL_RE_INTENT = re.compile(
    r"\b(e-?mail|send\s+(an\s+)?e-?mail|send\s+mail|mail\s+(this|it|me|my))\b",
    re.I,
)
ALLOWED_TOOLS = {"spending_stat", "search_notes", "gold_price", "send_email"}
SAFE_NUMBERS = set(range(1, 13)) | set(range(2014, 2027))


class FinanceReport(BaseModel):
    income: float = Field(..., description="Income total from a tool, or 0 if unknown")
    expense: float = Field(..., description="Expense total from a tool, or 0 if unknown")
    source_tool: str = Field(..., description="Which tool produced the numbers")


def wants_email(text: str) -> bool:
    """True only for a real email request, not the word spend or family."""
    return bool(EMAIL_RE_INTENT.search(text or ""))


def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text or "")
    return match.group(0) if match else None


def check_email_request(question: str, to_address: str | None = None) -> dict[str, Any]:
    address = extract_email(to_address or "") or extract_email(question)
    if wants_email(question) and not address:
        return {
            "ok": False,
            "rule": "email_requires_address",
            "message": "Guardrail: email was requested but no address was given. Refusing to send mail.",
        }
    if to_address and not extract_email(to_address):
        return {
            "ok": False,
            "rule": "email_requires_address",
            "message": "Guardrail: refuse email unless a valid address is given.",
        }
    return {"ok": True, "rule": "email_requires_address", "address": address}


def allowed_amounts(research_stats: dict[str, Any] | None, rag_hits: list[dict[str, Any]] | None) -> set[float]:
    allowed: set[float] = set()
    stats = research_stats or {}
    for key in ("income_total", "expense_total", "transfer_total", "unknown_total", "total_amount", "price", "row_count"):
        if stats.get(key) is not None:
            try:
                allowed.add(round(float(stats[key]), 2))
            except (TypeError, ValueError):
                pass
    for row in stats.get("by_category") or []:
        try:
            allowed.add(round(float(row.get("amount") or 0), 2))
        except (TypeError, ValueError):
            pass
    for hit in rag_hits or []:
        try:
            allowed.add(round(float(hit.get("amount") or 0), 2))
        except (TypeError, ValueError):
            pass
    allowed.add(0.0)
    return allowed


def invented_amounts(text: str, allowed: set[float]) -> list[float]:
    invented: list[float] = []
    for match in NUMBER_RE.finditer(text or ""):
        value = float(match.group(1))
        if value in SAFE_NUMBERS or value.is_integer() and int(value) in SAFE_NUMBERS:
            continue
        if any(abs(value - known) < 0.051 for known in allowed):
            continue
        invented.append(value)
    return invented


def parse_structured(payload: dict[str, Any] | None) -> tuple[FinanceReport | None, list[str]]:
    issues: list[str] = []
    try:
        report = FinanceReport.model_validate(payload or {})
    except ValidationError as exc:
        return None, [f"Structured output invalid: {exc.errors()[0]['msg']}"]
    if report.source_tool not in ALLOWED_TOOLS:
        issues.append(f"source_tool must be one of {sorted(ALLOWED_TOOLS)}")
    return report, issues


def review_brief(
    question: str,
    draft: str,
    structured: dict[str, Any] | None,
    research_stats: dict[str, Any] | None,
    rag_hits: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    issues: list[str] = []
    email_check = check_email_request(question)
    if not email_check["ok"]:
        issues.append(email_check["message"])

    report, schema_issues = parse_structured(structured)
    issues.extend(schema_issues)

    allowed = allowed_amounts(research_stats, rag_hits)
    if report is not None:
        stats = research_stats or {}
        if "income_total" in stats and abs(float(report.income) - float(stats["income_total"])) > 0.05:
            issues.append("Do not invent amounts: income does not match spending_stat.")
        if "expense_total" in stats and abs(float(report.expense) - float(stats["expense_total"])) > 0.05:
            issues.append("Do not invent amounts: expense does not match spending_stat.")

    invented = invented_amounts(draft, allowed)
    if invented:
        issues.append(f"Do not invent amounts: {invented} were not in the tool/RAG evidence.")

    return {
        "ok": not issues,
        "issues": issues,
        "structured": report.model_dump() if report else structured,
        "email": email_check,
    }
