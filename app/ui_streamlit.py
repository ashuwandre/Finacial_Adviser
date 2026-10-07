from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv

from guardrails import wants_email
from multi_agent import FinanceTeam
from react_agent import RESEARCHER_PROMPT, FinanceReactAgent

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
SESSION = requests.Session()
SESSION.trust_env = False


def backend_url() -> str:
    import os

    return os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def backend_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{backend_url()}{path}"
    last_error = None
    for _ in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            time.sleep(0.6)
    raise last_error or requests.ConnectionError(f"Could not reach backend at {url}")


def backend_post(path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    response = SESSION.post(f"{backend_url()}{path}", json=data or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def safe_backend_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        return backend_get(path, params)
    except Exception as exc:
        st.sidebar.error(f"Backend {path} failed: {exc}")
        return None


@st.cache_resource(show_spinner="Starting the financial assistant...")
def get_researcher() -> FinanceReactAgent:
    return FinanceReactAgent(prompt=RESEARCHER_PROMPT)


st.set_page_config(page_title="Financial Adviser", page_icon="💰", layout="wide")
st.title("Financial Adviser")
st.caption("RAG + LLM over 3 Excel files: yearly tracker, total expenditure, Krisala Cosmo.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "pending_hitl" not in st.session_state:
    st.session_state.pending_hitl = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask a question, for example: How much did I spend on food in 2018?",
            "trace": [],
        }
    ]
with st.sidebar:
    st.subheader("Status")
    health = safe_backend_get("/health")
    if health and health.get("status") == "ok":
        st.success(f"Backend OK · {health.get('rows', '?')} rows")
        files = health.get("files") or []
        st.caption(health.get("source") or "")
        if files:
            st.caption("Files: " + ", ".join(str(name) for name in files[:6]))
    else:
        st.warning("Start the FastAPI backend on port 8000 first.")

    filters = safe_backend_get("/filters") or {}
    years = filters.get("year_values") or []
    categories = filters.get("category_values") or []
    st.caption(f"Years: {years[0]}–{years[-1]}" if years else "No years loaded")
    st.caption(f"Categories: {len(categories)}")

    try:
        researcher = get_researcher()
        team = FinanceTeam(researcher)
    except Exception as exc:
        researcher = None
        team = None
        st.error(f"Agent/MCP not ready: {exc}")

    if st.button("Reload CSV"):
        safe_backend_get("/reload")
        st.rerun()

    st.divider()
    st.subheader("Expense data")
    expenses_dir = str(PROJECT_ROOT / "My Expenses")
    if st.button("Reload 3 Excel files"):
        backend_post("/set_csv", {"csv_path": expenses_dir})
        safe_backend_get("/reload")
        st.success("Reloaded yearly tracker, total expenditure, and Krisala")
        st.rerun()
    uploaded = st.file_uploader("Upload any CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix.lower() or ".csv"
        save_path = UPLOADS_DIR / f"uploaded_{int(time.time())}{suffix}"
        save_path.write_bytes(uploaded.getvalue())
        backend_post("/set_csv", {"csv_path": str(save_path)})
        safe_backend_get("/reload")
        st.success(f"Loaded {save_path.name}")
        st.rerun()

    if st.button("New chat thread"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.pending_hitl = None
        st.session_state.last_metrics = {}
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "New memory thread. Previous turns will not be used.",
                "trace": [],
            }
        ]
        st.rerun()

    metrics = st.session_state.last_metrics or {}
    if metrics:
        st.divider()
        st.subheader("Last run")
        st.caption(f"Routing: {metrics.get('routing_via', '-')}")
        st.caption(f"Supervisor loops: {metrics.get('supervisor_loops', 0)}")
        st.caption(f"Specialist calls: {metrics.get('specialist_calls', 0)}")
        st.caption(f"Retries: {metrics.get('retries', 0)}")
        st.caption(f"Elapsed: {metrics.get('elapsed_ms', 0)} ms")


def _store_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    payload["content"] = result.get("answer") or ""
    payload["team"] = result.get("team") or []
    payload["trace"] = result.get("trace") or []
    question_text = (result.get("pending_hitl") or {}).get("question") or ""
    allow_hitl = bool(result.get("pending_hitl")) and wants_email(question_text)
    payload["needs_approval"] = allow_hitl
    st.session_state.last_metrics = result.get("metrics") or {}
    st.session_state.pending_hitl = result.get("pending_hitl") if allow_hitl else None


def _resume_hitl(decision: str) -> None:
    pending = st.session_state.pending_hitl or {}
    payload = {"role": "assistant", "content": "", "trace": [], "team": [], "needs_approval": False}
    if team is None:
        payload["content"] = "The assistant could not start. Check OPENAI_API_KEY in app/.env."
        st.session_state.pending_hitl = None
        st.session_state.messages.append(payload)
        st.rerun()
        return
    try:
        result = team.ask(
            pending.get("question") or "",
            thread_id=st.session_state.thread_id,
            hitl_decision=decision,
            resume=pending,
        )
        _store_result(result, payload)
        if not result.get("pending_hitl"):
            st.session_state.pending_hitl = None
            payload["needs_approval"] = False
    except Exception as exc:
        payload["content"] = f"Could not continue after {decision}: {exc}"
        st.session_state.pending_hitl = None
    st.session_state.messages.append(payload)
    st.rerun()


waiting = bool(st.session_state.pending_hitl)
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        last = index == len(st.session_state.messages) - 1
        if last and waiting and message.get("needs_approval") and team is not None:
            st.warning("Email is paused. `send_email` runs only if you approve.")
            left, right = st.columns(2)
            if left.button("Approve email", type="primary", key="hitl_approve"):
                _resume_hitl("approve")
            if right.button("Reject email", key="hitl_reject"):
                _resume_hitl("reject")

if waiting:
    st.sidebar.divider()
    st.sidebar.warning("Waiting for email approval")
    if st.sidebar.button("Approve email", key="hitl_approve_sidebar", type="primary"):
        _resume_hitl("approve")
    if st.sidebar.button("Reject email", key="hitl_reject_sidebar"):
        _resume_hitl("reject")

try:
    question = st.chat_input(
        "Approve or reject the email first" if waiting else "Ask about your spending",
        disabled=waiting,
    )
except TypeError:
    question = st.chat_input("Ask about your spending")

if question and waiting:
    st.warning("Choose Approve email or Reject email before asking a new question.")
elif question:
    st.session_state.messages.append({"role": "user", "content": question, "trace": []})
    payload = {
        "role": "assistant",
        "content": "",
        "trace": [],
        "team": [],
        "needs_approval": False,
    }
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if team is None:
            payload["content"] = "The assistant could not start. Check OPENAI_API_KEY in app/.env."
            st.markdown(payload["content"])
        else:
            try:
                result = team.ask(question, thread_id=st.session_state.thread_id)
                _store_result(result, payload)
                st.markdown(payload["content"])
            except Exception as exc:
                payload["content"] = f"Sorry, I could not answer that. {exc}"
                st.markdown(payload["content"])
        st.session_state.messages.append(payload)
        if payload.get("needs_approval"):
            st.rerun()
