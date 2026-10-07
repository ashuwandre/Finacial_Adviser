"""ReAct agent that uses MCP tools.

Concepts used here:
- ReAct loop (Thought -> Act -> Observe) via create_react_agent
- MCP tool discovery via MultiServerMCPClient
- Short-term memory via a checkpointer + thread_id
- System prompt / role instructions
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

SYSTEM_PROMPT = """You are a careful household financial adviser.

Use tools instead of guessing numbers:
- spending_stat for income, expense, category, year, or month questions
- search_notes to retrieve real transaction notes (RAG), not only totals
- gold_price for live gold price
- send_email only when the user asked to email a report and provided an address

Never invent amounts. If a category is unknown, say so.
"""

RESEARCHER_PROMPT = """You are the researcher on a household money-brief team.

Business need: gather evidence for a short weekly money brief.

Always use tools. Do not write a polished summary.
1. If the user asks about gold or metal price, call gold_price only. Do not call spending_stat.
2. Call spending_stat for income, expense, category, year, or month questions.
3. Call search_notes for Netflix, train, grocery, or other note searches.
4. Call send_email only if the user gave an email address.

Return the raw tool facts. Never invent income or expense numbers.
"""

SUPERVISOR_PROMPT = """You are the supervisor of a household-finance team.
Decide the next graph node. You may run more than one specialist across loops.
- gold: live gold / metal / XAU price, or whether they can buy gold
- notes: Excel RAG (send money, family support, 2025 totals, savings)
- spend: income totals only when notes are not enough
- email: send a summary (only after totals exist and the human approved)
Rules:
- Compound questions (gold AND Excel, e.g. gold price + send money 2025) need gold and notes, one per loop.
- Do not skip writer if the user asked more than gold price. Writer must answer every part.
- Skip writer (next_node=end) only for a gold-price-only question after gold_price returned.
- If email is requested, an address exists, totals are ready, and HITL is not decided: next_node=hitl.
- If HITL_APPROVED, next specialist is email.
- If HITL_REJECTED, next_node=end.
- Otherwise after facts are ready, next_node=writer.
"""

SPEND_PROMPT = """You are the spend specialist. You have only spending_stat.
Always call spending_stat with year if the user gave a year.
Also pass query as the topic words from the question (example: family support).
Do not use a strict category name if the Excel note might be different.
Return raw tool facts. Never invent numbers.
"""

NOTES_PROMPT = """You are the Excel RAG specialist. You have only search_notes.
Call search_notes for each topic in the user question (letter case does not matter).
Pass year when given. Return retrieved Excel line items so they can be added up.
Never invent amounts.
"""

GOLD_PROMPT = """You are the gold specialist. You have only gold_price.
Call gold_price. Do not call any spending tool. Return the tool price.
"""

EMAIL_PROMPT = """You are the email specialist. You have spending_stat and send_email.
If the message does not contain HITL_APPROVED: call spending_stat only. Do not call send_email.
If the message contains HITL_APPROVED: call send_email with the user address, subject
Your spending summary, and a body that uses only tool totals. Never invent numbers.
"""


def _checkpointer():
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def _create_react_agent(model, tools, checkpointer, prompt: str = SYSTEM_PROMPT):
    from langgraph.prebuilt import create_react_agent

    try:
        return create_react_agent(
            model,
            tools,
            prompt=prompt,
            checkpointer=checkpointer,
        )
    except TypeError:
        return create_react_agent(
            model,
            tools,
            state_modifier=prompt,
            checkpointer=checkpointer,
        )


def _clean_env(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip().strip("\"'")
        if value and value.strip("# ").strip():
            return value
    return ""


def make_llm() -> ChatOpenAI | AzureChatOpenAI:
    openai_key = _clean_env("OPENAI_API_KEY")
    azure_key = _clean_env("AZURE_OPENAI_API_KEY")
    # A normal OpenAI key starts with sk- and does not use an Azure endpoint.
    if not openai_key and azure_key.startswith("sk-"):
        openai_key = azure_key

    if openai_key:
        model = _clean_env("OPENAI_MODEL", "AZURE_OPENAI_MODEL_NAME") or "gpt-4o-mini"
        return ChatOpenAI(api_key=openai_key, model=model, temperature=0)

    endpoint = _clean_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_BASE")
    deployment = _clean_env("AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_MODEL_NAME")
    api_version = _clean_env("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
    if not azure_key or not endpoint or not deployment:
        raise ValueError(
            "Add OPENAI_API_KEY=sk-... to app/.env for a normal OpenAI key. "
            "Leave AZURE_OPENAI_ENDPOINT empty. You do not need an Azure URL."
        )

    host = endpoint.lower()
    if "your-resource" in host or "example.com" in host or "xxxx" in host:
        raise ValueError(
            "You are using a placeholder Azure URL. For a normal OpenAI key, set "
            "OPENAI_API_KEY=sk-your-key and remove AZURE_OPENAI_ENDPOINT."
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_key=azure_key,
        api_version=api_version,
        temperature=0,
    )


def _mcp_port() -> int:
    return int(os.getenv("MCP_PORT", "8001"))


def _mcp_url() -> str:
    return os.getenv("MCP_URL", f"http://127.0.0.1:{_mcp_port()}/mcp")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def ensure_mcp_http() -> None:
    """Start the MCP HTTP server if Streamlit/Windows stdio would hang."""
    port = _mcp_port()
    if _port_open(port):
        return
    log_path = APP_DIR / "mcp_http.log"
    log_file = open(log_path, "a", encoding="utf-8")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, str(APP_DIR / "mcp_server.py"), "--http"],
        cwd=str(APP_DIR),
        env={**os.environ, "BACKEND_URL": os.getenv("BACKEND_URL", "http://127.0.0.1:8000"), "MCP_PORT": str(port)},
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags,
    )
    for _ in range(20):
        if _port_open(port):
            return
        time.sleep(0.25)
    raise RuntimeError(f"MCP HTTP server did not start on port {port}. See {log_path}")


def make_mcp_client() -> MultiServerMCPClient:
    ensure_mcp_http()
    return MultiServerMCPClient(
        {
            "finance": {
                "transport": "streamable_http",
                "url": _mcp_url(),
            }
        }
    )


def _run(coro, timeout: int = 180):
    """Run async MCP calls on a fresh loop. Streamlit's loop breaks stdio/anyio."""
    def runner():
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result(timeout=timeout)


def _sync_mcp_tools(tools: list[Any]) -> list[Any]:
    """MCP tools are async-only; LangGraph sync invoke needs a func wrapper."""
    wrapped: list[Any] = []
    for tool in tools:
        if getattr(tool, "func", None) is not None:
            wrapped.append(tool)
            continue

        def _bind(current):
            def _func(**kwargs):
                return _run(current.ainvoke(kwargs))

            async def _afunc(**kwargs):
                return await current.ainvoke(kwargs)

            return StructuredTool.from_function(
                name=current.name,
                description=current.description or current.name,
                func=_func,
                coroutine=_afunc,
                args_schema=getattr(current, "args_schema", None),
            )

        wrapped.append(_bind(tool))
    return wrapped


def extract_react_trace(messages: list[Any]) -> list[dict[str, Any]]:
    """Turn LangGraph messages into Thought / Act / Observe / Answer steps."""
    steps: list[dict[str, Any]] = []
    for message in messages:
        message_type = getattr(message, "type", "") or message.__class__.__name__.lower()
        content = getattr(message, "content", "") or ""
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            if content:
                steps.append({"phase": "Thought", "detail": str(content)})
            else:
                steps.append({"phase": "Thought", "detail": "Model chose a tool."})
            for call in tool_calls:
                steps.append(
                    {
                        "phase": "Act",
                        "detail": f"{call.get('name')}({call.get('args', {})})",
                    }
                )
            continue

        if message_type in {"tool", "toolmessage"} or type(message).__name__ == "ToolMessage":
            name = getattr(message, "name", "tool")
            preview = str(content)
            if len(preview) > 400:
                preview = preview[:400] + "..."
            steps.append({"phase": "Observe", "detail": f"{name} -> {preview}"})
            continue

        if message_type in {"ai", "aimessage"} and content:
            steps.append({"phase": "Answer", "detail": str(content)})

    return steps


class FinanceReactAgent:
    def __init__(
        self,
        prompt: str = SYSTEM_PROMPT,
        *,
        client: MultiServerMCPClient | None = None,
        tools: list[Any] | None = None,
        llm: ChatOpenAI | AzureChatOpenAI | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.client = client or make_mcp_client()
        loaded = tools if tools is not None else _sync_mcp_tools(_run(self.client.get_tools()))
        if allowed_tools:
            allow = set(allowed_tools)
            loaded = [tool for tool in loaded if getattr(tool, "name", "") in allow]
        self.tools = loaded
        self.llm = llm or make_llm()
        self.checkpointer = _checkpointer()
        self.graph = _create_react_agent(self.llm, self.tools, self.checkpointer, prompt)

    def tool_names(self) -> list[str]:
        return [getattr(tool, "name", str(tool)) for tool in self.tools]

    def ask(self, query: str, thread_id: str = "study-thread") -> dict[str, Any]:
        payload = {"messages": [{"role": "user", "content": query}]}
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = _run(self.graph.ainvoke(payload, config))
        except Exception:
            result = self.graph.invoke(payload, config)
        messages = result.get("messages", [])
        answer = ""
        if messages:
            answer = getattr(messages[-1], "content", "") or str(messages[-1])
        return {
            "answer": answer,
            "trace": extract_react_trace(messages),
            "messages": messages,
            "tools": self.tool_names(),
        }
