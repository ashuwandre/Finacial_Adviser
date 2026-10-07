"""MCP server for Financial Adviser tools.

Study note:
The ReAct agent does not import these functions directly.
It discovers them through MCP (list_tools / call_tool).
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from guardrails import check_email_request, extract_email

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
SESSION = requests.Session()
SESSION.trust_env = False

MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
mcp = FastMCP("financial-adviser", host="127.0.0.1", port=MCP_PORT)


def _backend_get(path: str, params: dict | None = None) -> dict:
    response = SESSION.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _compact(params: dict) -> dict:
    return {key: value for key, value in params.items() if value is not None and value != ""}


def fetch_gold_spot() -> dict:
    """Live gold price. Tries more than one public API because metals.live is often down."""
    try:
        data = SESSION.get("https://api.gold-api.com/price/XAU", timeout=5).json()
        if data.get("price") is not None:
            return {
                "price": data.get("price"),
                "currency": data.get("currency") or "USD",
                "source": "gold-api",
            }
    except Exception:
        pass
    try:
        data = SESSION.get("https://api.metals.live/v1/spot/gold", timeout=5).json()
        spot = data[0] if isinstance(data, list) else data
        if isinstance(spot, dict) and spot.get("price") is not None:
            return {
                "price": spot.get("price"),
                "currency": spot.get("currency") or "USD",
                "source": "metals.live",
            }
    except Exception as exc:
        return {"ok": False, "error": f"Could not fetch gold price: {exc}"}
    return {"ok": False, "error": "Could not fetch gold price from public APIs."}


@mcp.tool()
def spending_stat(
    year: int | None = None,
    month: int | None = None,
    direction: str | None = None,
    category: str | None = None,
    query: str | None = None,
) -> str:
    """Get household spending statistics from the FastAPI backend.

    Use this for income, expense, category, year, or month questions.
    Pass query as the user's topic words (for example family support) so
    matching is by note text, not only an exact category name.
    """
    stats = _backend_get(
        "/stats",
        _compact(
            {
                "year": year,
                "month": month,
                "direction": direction,
                "category": category,
                "q": query,
            }
        ),
    )
    return json.dumps(stats, default=str)


@mcp.tool()
def search_notes(
    query: str,
    k: int = 12,
    year: int | None = None,
    category: str | None = None,
) -> str:
    """RAG search over the three Excel files (yearly tracker, total expenditure, Krisala).
    Use this for any question about the workbooks, including family support, savings, or flat cost.
    """
    result = _backend_get(
        "/search_notes",
        _compact({"q": query, "k": k, "year": year, "category": category}),
    )
    return json.dumps(result, default=str)


def send_finance_email(to: str, subject: str, body: str) -> dict:
    gate = check_email_request(f"{to} {subject} {body}", to_address=to)
    if not gate["ok"]:
        return {"ok": False, "error": gate["message"]}
    if not extract_email(to):
        return {"ok": False, "error": "Guardrail: refuse email unless a valid address is given."}
    smtp_server = (os.getenv("SMTP_SERVER") or os.getenv("SMTP_HOST") or "").strip()
    smtp_port = int((os.getenv("SMTP_PORT") or "587").strip())
    smtp_user = (os.getenv("SMTP_USER") or "").strip()
    smtp_password = (os.getenv("SMTP_PASSWORD") or "").replace(" ", "").strip()
    mail_from = (os.getenv("EMAIL_FROM") or os.getenv("EMIL_FROM") or smtp_user).strip()
    email_mode = (os.getenv("EMAIL_MODE") or "local").strip().lower()
    use_local = email_mode != "smtp" or len(smtp_password) != 16
    if use_local:
        inbox = APP_DIR / "sent_emails"
        inbox.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_to = "".join(ch if ch.isalnum() or ch in "@._-" else "_" for ch in to)
        path = inbox / f"{stamp}_{safe_to}.txt"
        path.write_text(f"To: {to}\nSubject: {subject}\n\n{body}\n", encoding="utf-8")
        return {
            "ok": True,
            "message": (
                f"Saved locally for {to} (study mode, no Gmail needed). "
                f"Open {path.name} in app/sent_emails."
            ),
        }
    if not all([smtp_server, smtp_user, smtp_password, mail_from]):
        return {"ok": False, "error": "SMTP settings are missing in .env"}
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = mail_from
    message["To"] = to

    def _send(use_ssl: bool, port: int) -> None:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, port, timeout=30) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(mail_from, [to], message.as_string())
            return
        with smtplib.SMTP(smtp_server, port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(mail_from, [to], message.as_string())

    try:
        _send(use_ssl=smtp_port == 465, port=smtp_port)
    except Exception:
        try:
            _send(use_ssl=True, port=465)
        except Exception as exc:
            return {
                "ok": False,
                "error": (
                    "Could not send email. Use SMTP_HOST=smtp.gmail.com and a Gmail App Password. "
                    f"Details: {exc}"
                ),
            }
    return {"ok": True, "message": f"Email sent to {to}"}


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Email a financial summary. Call only when the user asked to send mail and gave an address."""
    return json.dumps(send_finance_email(to, subject, body), default=str)


@mcp.tool()
def gold_price() -> str:
    """Fetch today's live gold spot price. Use this for any gold / metal price question."""
    return json.dumps(fetch_gold_spot(), default=str)


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # Separate port from FastAPI (8000) so Streamlit can talk to MCP over HTTP.
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
