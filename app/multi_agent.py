"""Supervisor loop: specialist handoff, optional writer/reviewer, HITL email.

Code owns the graph. The LLM proposes the next node. If the LLM fails,
keyword routing still picks specialists. Email waits for human approval.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from agent_log import log_event
from excel_rag import line_items_for_lookups, merge_hits, search_excel_docs, sum_amounts
from guardrails import (
    FinanceReport,
    check_email_request,
    extract_email,
    review_brief,
    wants_email,
)
from react_agent import (
    EMAIL_PROMPT,
    GOLD_PROMPT,
    NOTES_PROMPT,
    RESEARCHER_PROMPT,
    SPEND_PROMPT,
    SUPERVISOR_PROMPT,
    FinanceReactAgent,
)

SPECIALISTS = ("spend", "notes", "gold", "email")
MAX_SUPERVISOR_LOOPS = 6
SPECIALIST_TIMEOUT_SEC = 90
SPECIALIST_RETRIES = 1


class TeamState(TypedDict, total=False):
    question: str
    thread_id: str
    route: str
    next_node: str
    routing_via: str
    completed_specialists: list[str]
    supervisor_loops: int
    research_text: str
    research_stats: dict[str, Any]
    rag_hits: list[dict[str, Any]]
    react_trace: list[dict[str, Any]]
    draft: str
    structured: dict[str, Any]
    review_ok: bool
    review_issues: list[str]
    revisions: int
    answer: str
    agent_trace: list[dict[str, str]]
    email_gate: dict[str, Any]
    hitl_decision: str
    pending_hitl: dict[str, Any]
    metrics: dict[str, Any]


class SupervisorPlan(BaseModel):
    next_specialist: Literal["spend", "notes", "gold", "email", "none"] = "none"
    next_node: Literal["specialist", "writer", "hitl", "end"] = "writer"
    skip_writer: bool = False
    needs_human: bool = False
    reason: str = ""


class ExcelLookupPlan(BaseModel):
    year: int | None = None
    lookups: list[str] = Field(default_factory=list)
    want_sum: bool = False


def _unwrap_tool_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _parse_json_blob(text: Any) -> dict[str, Any] | None:
    raw = _unwrap_tool_content(text)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("text"):
            return _parse_json_blob(data[0]["text"])
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _is_gold_question(question: str) -> bool:
    text = (question or "").lower()
    if any(
        phrase in text
        for phrase in ("gold price", "price of gold", "buy gold", "metal price", "xau")
    ):
        return True
    if "gold" not in text:
        return False
    return not _wants_excel(text)


def _wants_excel(question: str) -> bool:
    text = (question or "").lower()
    if re.search(r"\b20\d{2}\b", text):
        return True
    return any(
        word in text
        for word in (
            "spend",
            "spent",
            "expense",
            "income",
            "total",
            "find",
            "send",
            "saving",
            "family",
            "excel",
        )
    )


def _needed_specialists(question: str) -> list[str]:
    needed: list[str] = []
    text = question or ""
    if _is_gold_question(text):
        needed.append("gold")
    if wants_email(text):
        if "notes" not in needed:
            needed.append("notes")
        needed.append("email")
        return needed
    if _wants_excel(text) or "gold" not in needed:
        needed.append("notes")
    if "income" in text.lower() and "spend" not in needed:
        needed.append("spend")
    return needed or ["notes"]


def _collect_tool_payloads(messages: list[Any]) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
    stats: dict[str, Any] = {}
    source = None
    rag_hits: list[dict[str, Any]] = []
    for message in messages:
        if type(message).__name__ != "ToolMessage" and getattr(message, "type", "") != "tool":
            continue
        name = getattr(message, "name", "") or ""
        payload = _parse_json_blob(getattr(message, "content", ""))
        if not payload:
            continue
        if name == "spending_stat":
            stats = {**stats, **payload}
            source = source or "spending_stat"
        elif name == "gold_price":
            source = source or "gold_price"
            stats = {**stats, **payload}
        elif name == "search_notes":
            rag_hits = payload.get("hits") or rag_hits
            source = source or "search_notes"
        elif name == "send_email":
            stats["email_result"] = payload
            source = "send_email"
    return stats, source, rag_hits


def _fallback_plan(state: TeamState) -> SupervisorPlan:
    question = state.get("question") or ""
    completed = list(state.get("completed_specialists") or [])
    stats = state.get("research_stats") or {}
    hits = state.get("rag_hits") or []
    decision = (state.get("hitl_decision") or "").lower()
    needed = _needed_specialists(question)
    pending = [name for name in needed if name != "email" and name not in completed]

    if decision == "reject":
        return SupervisorPlan(next_node="end", skip_writer=True, reason="Human rejected email.")
    if decision == "approve" and "email" in needed and "email" not in completed:
        return SupervisorPlan(next_specialist="email", next_node="specialist", reason="Human approved email.")
    if pending:
        return SupervisorPlan(next_specialist=pending[0], next_node="specialist", reason="Code queue.")
    if (
        "email" in needed
        and extract_email(question)
        and decision not in {"approve", "reject"}
        and not stats.get("email_result")
    ):
        return SupervisorPlan(next_node="hitl", needs_human=True, reason="Email waits for human.")
    if "email" in needed and not extract_email(question):
        return SupervisorPlan(next_node="writer", reason="Email asked but no address.")

    gold_only = needed == ["gold"] and stats.get("price") is not None
    if gold_only:
        return SupervisorPlan(next_node="end", skip_writer=True, reason="Gold price is enough.")
    return SupervisorPlan(next_node="writer", reason="LLM answers from Excel RAG excerpts.")


def _plan_from_llm(state: TeamState, llm) -> SupervisorPlan | None:
    try:
        picked = llm.with_structured_output(SupervisorPlan).invoke(
            [
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": state.get("question"),
                            "completed_specialists": state.get("completed_specialists") or [],
                            "has_totals": bool((state.get("research_stats") or {}).get("expense_total") is not None
                            or (state.get("research_stats") or {}).get("income_total")),
                            "has_gold": (state.get("research_stats") or {}).get("price") is not None,
                            "notes_count": len(state.get("rag_hits") or []),
                            "hitl_decision": state.get("hitl_decision") or "",
                            "email_sent": bool((state.get("research_stats") or {}).get("email_result")),
                            "loops": state.get("supervisor_loops") or 0,
                        },
                        default=str,
                    ),
                },
            ]
        )
        if picked.next_node == "specialist" and picked.next_specialist not in SPECIALISTS:
            return None
        return picked
    except Exception:
        return None


def _merge_metrics(state: TeamState, **updates: Any) -> dict[str, Any]:
    metrics = dict(state.get("metrics") or {})
    metrics.update(updates)
    return metrics


def supervisor_node(state: TeamState, llm) -> dict[str, Any]:
    loops = int(state.get("supervisor_loops") or 0) + 1
    thread_id = state.get("thread_id") or ""
    question = state.get("question") or ""
    fallback = _fallback_plan(state)
    plan = fallback
    via = "fallback"
    pending = [
        name
        for name in _needed_specialists(question)
        if name != "email" and name not in (state.get("completed_specialists") or [])
    ]

    if loops > MAX_SUPERVISOR_LOOPS:
        plan = SupervisorPlan(next_node="writer" if not fallback.skip_writer else "end", reason="Loop cap.")
        via = "fallback"

    decision = (state.get("hitl_decision") or "").lower()
    if decision == "reject":
        plan = SupervisorPlan(next_node="end", skip_writer=True, reason="Human rejected email.")
        via = "hitl"
    elif decision == "approve" and wants_email(question) and extract_email(question):
        plan = SupervisorPlan(next_specialist="email", next_node="specialist", reason="Human approved email.")
        via = "hitl"
    elif fallback.next_node == "hitl" and wants_email(question) and extract_email(question):
        plan = fallback
        via = "fallback"
    elif plan.next_node == "hitl" or plan.next_specialist == "email":
        if not (wants_email(question) and extract_email(question) and decision == "approve"):
            plan = fallback if fallback.next_node != "hitl" else SupervisorPlan(
                next_specialist="spend", next_node="specialist", reason="Not an email question."
            )
            via = "fallback"

    answer = state.get("answer") or ""
    if plan.next_node == "end":
        if (state.get("hitl_decision") or "").lower() == "reject":
            answer = "Email was not sent. You rejected it."
        elif not answer:
            answer = _plain_reply(
                question,
                state.get("research_stats") or {},
                state.get("research_text") or "",
                state.get("rag_hits") or [],
            )

    log_event(
        "supervisor",
        thread_id=thread_id,
        agent="supervisor",
        detail=f"{via} -> {plan.next_node}/{plan.next_specialist}: {plan.reason}",
        extra={"via": via, "next_node": plan.next_node, "specialist": plan.next_specialist},
    )
    return {
        "route": plan.next_specialist,
        "next_node": plan.next_node,
        "routing_via": via,
        "supervisor_loops": loops,
        "answer": answer,
        "metrics": _merge_metrics(state, routing_via=via, supervisor_loops=loops, fallback_used=via == "fallback"),
        "agent_trace": (state.get("agent_trace") or [])
        + [{"agent": "supervisor", "detail": f"{via}: {plan.next_node} {plan.next_specialist} — {plan.reason}"}],
    }


def _ask_specialist(agent: FinanceReactAgent, question: str, thread_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(SPECIALIST_RETRIES + 1):
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(agent.ask, question, thread_id).result(timeout=SPECIALIST_TIMEOUT_SEC)
            elapsed = int((time.perf_counter() - started) * 1000)
            log_event(
                "specialist_ok",
                thread_id=thread_id,
                agent="specialist",
                detail=f"attempt={attempt + 1} {elapsed}ms",
                extra={"elapsed_ms": elapsed, "attempt": attempt + 1},
            )
            return result
        except Exception as exc:
            last_error = exc
            metrics["retries"] = int(metrics.get("retries") or 0) + 1
            log_event(
                "specialist_retry",
                thread_id=thread_id,
                agent="specialist",
                detail=str(exc),
                ok=False,
                extra={"attempt": attempt + 1},
            )
    return {
        "answer": f"Specialist failed: {last_error}",
        "messages": [],
        "trace": [{"phase": "Observe", "detail": str(last_error)}],
    }


def _fallback_lookups(question: str) -> ExcelLookupPlan:
    year = None
    found = re.search(r"\b(20\d{2})\b", question or "")
    if found:
        year = int(found.group(1))
    cleaned = re.sub(r"\b(20\d{2})\b", " ", question or "", flags=re.I)
    cleaned = re.sub(
        r"\b(how|much|did|do|i|we|my|the|a|an|on|in|for|to|of|spend|spent|expense|expenses|total|sum|combined|together|check|today|gold|price|xau|metal|also|whether|can|buy|find|please)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    parts = [part.strip(" .?") for part in re.split(r",|/|&|\band\b|\bplus\b|\bas well as\b", cleaned, flags=re.I)]
    lookups = [part for part in parts if len(part) > 1]
    want_sum = bool(re.search(r"\b(total|sum|combined|together|plus)\b", question or "", flags=re.I))
    return ExcelLookupPlan(year=year, lookups=lookups or [question], want_sum=want_sum)


def _excel_lookup_plan(llm, question: str) -> ExcelLookupPlan:
    fallback = _fallback_lookups(question)
    try:
        picked = llm.with_structured_output(ExcelLookupPlan).invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract search phrases to look up in an expense spreadsheet. "
                        "If the question has several parts, extract only the Excel parts "
                        "(for example send money, family support). Ignore live gold price wording. "
                        "Keep the user's own words and add only grammatical variants "
                        "(singular/plural, dine/dining, send/sent). "
                        "If they ask whether they can buy gold, also look up saving/savings/investment. "
                        "Do not invent extra categories. Ignore letter case. "
                        "If they want one combined total, set want_sum true."
                    ),
                },
                {"role": "user", "content": question},
            ]
        )
        lookups = [item.strip() for item in (picked.lookups or []) if str(item).strip()]
        return ExcelLookupPlan(
            year=picked.year or fallback.year,
            lookups=lookups or fallback.lookups,
            want_sum=bool(picked.want_sum or fallback.want_sum),
        )
    except Exception:
        return fallback


def specialist_node(state: TeamState, agents: dict[str, FinanceReactAgent]) -> dict[str, Any]:
    question = state.get("question") or ""
    route = state.get("route") or "spend"
    if route not in agents:
        route = "spend"
    thread_id = f"{state.get('thread_id') or 'team-thread'}-{route}"
    asked = question
    if route == "email" and (state.get("hitl_decision") or "").lower() == "approve":
        asked = f"{question}\nHITL_APPROVED: send the email now using send_email."
    elif route == "email":
        asked = f"{question}\nDo not call send_email. Collect totals only."

    metrics = _merge_metrics(state)
    metrics["specialist_calls"] = int(metrics.get("specialist_calls") or 0) + 1
    if route == "gold":
        try:
            from mcp_server import fetch_gold_spot

            gold = fetch_gold_spot()
        except Exception as exc:
            gold = {"ok": False, "error": str(exc)}
        stats = {**(state.get("research_stats") or {}), **gold}
        stats["source_tool"] = "gold_price"
        completed = list(state.get("completed_specialists") or [])
        if "gold" not in completed:
            completed.append("gold")
        detail = f"gold_price={gold.get('price')} {gold.get('currency') or ''}"
        log_event("gold", thread_id=thread_id, agent="gold_specialist", detail=detail)
        return {
            "research_text": detail,
            "research_stats": stats,
            "rag_hits": state.get("rag_hits") or [],
            "react_trace": (state.get("react_trace") or [])
            + [{"phase": "Act", "detail": "gold_price()"}],
            "completed_specialists": completed,
            "email_gate": check_email_request(question),
            "metrics": metrics,
            "agent_trace": (state.get("agent_trace") or [])
            + [{"agent": "gold_specialist", "detail": detail}],
        }
    if route == "notes":
        plan = _excel_lookup_plan(agents[route].llm, question)
        hit_lists = [search_excel_docs(item, k=8, year=plan.year) for item in plan.lookups]
        if not plan.lookups:
            hit_lists.append(search_excel_docs(question, k=10, year=plan.year))
        merged = merge_hits(hit_lists)
        lines = line_items_for_lookups(merged, plan.lookups)
        total = sum_amounts(lines) if lines and (plan.want_sum or len(plan.lookups) <= 1) else None
        stats = {**(state.get("research_stats") or {})}
        stats.update(
            {
                "source_tool": "search_notes",
                "hits": lines or merged,
                "excel_lines": [
                    {
                        "note": hit.get("note"),
                        "category": hit.get("category"),
                        "year": hit.get("year") or hit.get("sheet"),
                        "amount": hit.get("amount"),
                        "file": hit.get("file"),
                        "sheet": hit.get("sheet"),
                    }
                    for hit in lines
                ],
                "excel_total": total,
                "lookups": plan.lookups,
                "year": plan.year,
            }
        )
        if total is not None:
            stats["expense_total"] = total
        completed = list(state.get("completed_specialists") or [])
        if "notes" not in completed:
            completed.append("notes")
        detail = f"RAG lookups={plan.lookups} year={plan.year} lines={len(lines)} total={total}"
        log_event("excel_rag", thread_id=thread_id, agent="notes_specialist", detail=detail)
        return {
            "research_text": detail,
            "research_stats": stats,
            "rag_hits": lines or merged,
            "react_trace": (state.get("react_trace") or [])
            + [{"phase": "Act", "detail": f"search_notes({plan.lookups})"}],
            "completed_specialists": completed,
            "email_gate": check_email_request(question),
            "metrics": metrics,
            "agent_trace": (state.get("agent_trace") or [])
            + [{"agent": "notes_specialist", "detail": detail}],
        }

    result = _ask_specialist(agents[route], asked, thread_id, metrics)
    fresh, source, new_hits = _collect_tool_payloads(result.get("messages") or [])
    previous = state.get("research_stats") or {}
    stats = {**previous, **fresh}
    if previous.get("excel_lines") and float(fresh.get("expense_total") or 0) == 0:
        for key in ("excel_total", "excel_lines", "lookups", "hits", "expense_total"):
            if previous.get(key) is not None:
                stats[key] = previous[key]
        stats["source_tool"] = previous.get("source_tool") or "search_notes"
    rag_hits = new_hits or state.get("rag_hits") or []
    if source:
        stats["source_tool"] = source
    elif route == "gold":
        stats.setdefault("source_tool", "gold_price")
    elif route == "notes":
        stats.setdefault("source_tool", "search_notes")
    elif route == "email":
        stats.setdefault("source_tool", "send_email")
    else:
        stats.setdefault("source_tool", "spending_stat")

    completed = list(state.get("completed_specialists") or [])
    if route not in completed:
        completed.append(route)
    traces = list(state.get("react_trace") or [])
    traces.extend(result.get("trace") or [])
    name = f"{route}_specialist"
    trace = [{"agent": name, "detail": result.get("answer") or f"Used {route} MCP tools."}]
    for step in result.get("trace") or []:
        trace.append({"agent": name, "detail": f"{step['phase']}: {step['detail']}"})
        if step.get("phase") == "Act":
            log_event("tool", thread_id=thread_id, agent=name, detail=str(step.get("detail")))

    return {
        "research_text": result.get("answer") or state.get("research_text") or "",
        "research_stats": stats,
        "rag_hits": rag_hits,
        "react_trace": traces,
        "completed_specialists": completed,
        "email_gate": check_email_request(question),
        "metrics": metrics,
        "agent_trace": (state.get("agent_trace") or []) + trace,
    }


def _plain_reply(
    question: str,
    stats: dict[str, Any],
    researcher_text: str,
    rag_hits: list[dict[str, Any]] | None = None,
) -> str:
    lowered = (question or "").lower()
    parts: list[str] = []
    if stats.get("excel_total") is not None and stats.get("excel_lines"):
        lines = [
            f"- {item.get('note')}: ₹{float(item.get('amount') or 0):,.2f} ({item.get('sheet')})"
            for item in stats.get("excel_lines") or []
        ]
        parts.append(
            "Combined Excel total ₹{:,.2f}.\n{}".format(float(stats["excel_total"]), "\n".join(lines))
        )

    if stats.get("price") is not None:
        currency = stats.get("currency") or "USD"
        parts.append(f"Today's gold price is {stats['price']} {currency}.")
    elif _is_gold_question(question) and not parts:
        parts.append("I could not fetch today's gold price. Please try again in a moment.")

    hits = stats.get("hits") or rag_hits or []
    if hits:
        lines = ["Excel RAG matches:"]
        for hit in hits[:8]:
            excerpt = (hit.get("text") or hit.get("note") or "").strip()
            if len(excerpt) > 280:
                excerpt = excerpt[:280] + "..."
            amount = hit.get("amount")
            extra = f" ₹{float(amount):,.2f}" if amount else ""
            lines.append(f"- {excerpt}{extra}")
        parts.append("\n".join(lines))

    year = stats.get("year")
    category = stats.get("category")
    expense = stats.get("expense_total")
    income = stats.get("income_total")
    spend_asked = any(word in lowered for word in ("spend", "spent", "expense", "income", "total")) or wants_email(question)
    if spend_asked or (income is not None and expense is not None and not parts):
        if category and year is not None and expense is not None and "income" not in lowered:
            if float(expense) == 0:
                parts.append(
                    f"I found no matching {category} expenses in {year} "
                    "in the loaded Excel/CSV files."
                )
            else:
                parts.append(f"You spent ₹{float(expense):,.2f} on {category} in {year}.")
        elif year is not None and "income" in lowered and income is not None:
            parts.append(f"Your total income in {year} was ₹{float(income):,.2f}.")
        elif income is not None and expense is not None:
            if year is not None:
                parts.append(
                    f"In {year}, income was ₹{float(income):,.2f} "
                    f"and expenses were ₹{float(expense):,.2f}."
                )
            else:
                parts.append(
                    f"Income was ₹{float(income):,.2f} and expenses were ₹{float(expense):,.2f}."
                )

    mailed = stats.get("email_result") or {}
    if wants_email(question):
        address = extract_email(question)
        if not address:
            parts.append("I cannot email a report because no email address was given.")
        elif mailed.get("ok"):
            extra = mailed.get("message") or ""
            parts.append(f"Email step finished for {address}. {extra}")
        elif mailed:
            parts.append(mailed.get("error") or "I could not send the email.")

    if parts:
        return "\n\n".join(parts)
    text = (researcher_text or "").strip()
    if text:
        return text.split("\n")[0]
    return "I could not find those totals in the transaction data."


def _llm_from_excel(llm, question: str, hits: list[dict[str, Any]], stats: dict[str, Any] | None = None) -> str:
    stats = stats or {}
    lines = stats.get("excel_lines") or []
    total = stats.get("excel_total")
    line_text = "\n".join(
        f"- {item.get('note')} ({item.get('sheet')}): ₹{float(item.get('amount') or 0):,.2f}"
        for item in lines
    )
    computed = ""
    if total is not None:
        computed = f"Computed combined total from the Excel line items: ₹{float(total):,.2f}\n{line_text}"
    elif line_text:
        computed = f"Excel line items:\n{line_text}"
    gold_bit = ""
    if stats.get("price") is not None:
        gold_bit = f"Live gold price: {stats['price']} {stats.get('currency') or 'USD'}"
    context = "\n\n".join(
        f"[{hit.get('file')} / {hit.get('sheet')}]\n{hit.get('text') or hit.get('note')}"
        for hit in hits[:10]
    )
    message = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You answer every part of a compound household money question. "
                    "Use live gold price when provided. Use Excel excerpts for spend line items. "
                    "Letter case and word form do not matter. "
                    "If a computed combined total is provided, use that total and list each line. "
                    "If they ask whether they can buy gold, compare gold price with savings/"
                    "investment rows if present. Send money / money returned is already spent, "
                    "not cash in hand. Never invent amounts. If cash on hand is missing, say so. "
                    "Name file and sheet."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n{gold_bit}\n\n{computed}\n\nExcel excerpts:\n{context}"
                ),
            },
        ]
    )
    return str(getattr(message, "content", "") or message).strip()


def writer_node(state: TeamState, llm) -> dict[str, Any]:
    stats = state.get("research_stats") or {}
    hits = state.get("rag_hits") or stats.get("hits") or []
    email_gate = state.get("email_gate") or check_email_request(state["question"])
    structured = {
        "income": float(stats.get("income_total") or 0),
        "expense": float(stats.get("expense_total") or 0),
        "source_tool": stats.get("source_tool") or ("search_notes" if hits else "spending_stat"),
    }
    summary = ""
    if llm is not None and not wants_email(state["question"]) and (hits or stats.get("price") is not None):
        try:
            summary = _llm_from_excel(llm, state["question"], hits, stats)
        except Exception:
            summary = ""
    if not summary:
        summary = _plain_reply(
            state["question"],
            stats,
            state.get("research_text") or "",
            hits,
        )
    if not email_gate.get("ok") and wants_email(state["question"]):
        summary = email_gate.get("message") or summary
    log_event("writer", thread_id=state.get("thread_id") or "", agent="writer", detail=summary[:200])
    return {
        "draft": summary,
        "structured": structured,
        "agent_trace": (state.get("agent_trace") or [])
        + [{"agent": "writer", "detail": summary or json.dumps(structured)}],
    }


def reviewer_node(state: TeamState) -> dict[str, Any]:
    verdict = review_brief(
        question=state["question"],
        draft=state.get("draft") or "",
        structured=state.get("structured"),
        research_stats=state.get("research_stats"),
        rag_hits=state.get("rag_hits"),
    )
    revisions = int(state.get("revisions") or 0) + 1
    report = verdict.get("structured") or state.get("structured") or {}
    answer = (state.get("draft") or "").strip()
    log_event(
        "reviewer",
        thread_id=state.get("thread_id") or "",
        agent="reviewer",
        detail="Approved." if verdict["ok"] else "; ".join(verdict["issues"]),
        ok=bool(verdict["ok"]),
    )
    return {
        "review_ok": verdict["ok"],
        "review_issues": verdict["issues"],
        "revisions": revisions,
        "structured": report,
        "answer": answer,
        "agent_trace": (state.get("agent_trace") or [])
        + [
            {
                "agent": "reviewer",
                "detail": "Approved." if verdict["ok"] else "; ".join(verdict["issues"]),
            }
        ],
    }


def hitl_node(state: TeamState) -> dict[str, Any]:
    question = state.get("question") or ""
    stats = state.get("research_stats") or {}
    to = extract_email(question) or ""
    body = (
        f"Income: {stats.get('income_total')}\n"
        f"Expense: {stats.get('expense_total')}\n"
    )
    if stats.get("price") is not None:
        body += f"Gold: {stats.get('price')} {stats.get('currency') or 'USD'}\n"
    pending = {
        "to": to,
        "subject": "Your spending summary",
        "body": body,
        "question": question,
        "research_stats": stats,
        "rag_hits": state.get("rag_hits") or [],
        "research_text": state.get("research_text") or "",
        "completed_specialists": state.get("completed_specialists") or [],
        "agent_trace": state.get("agent_trace") or [],
        "react_trace": state.get("react_trace") or [],
        "metrics": state.get("metrics") or {},
    }
    answer = (
        f"Human approval needed before emailing {to or '(missing address)'}.\n\n"
        f"{body.strip()}\n\nUse Approve email or Reject email below."
    )
    log_event("hitl_wait", thread_id=state.get("thread_id") or "", agent="hitl", detail=f"to={to}")
    return {
        "pending_hitl": pending,
        "answer": answer,
        "draft": answer,
        "agent_trace": (state.get("agent_trace") or [])
        + [{"agent": "hitl", "detail": f"Waiting for human approval to email {to}"}],
    }


def _route_supervisor(state: TeamState) -> str:
    node = state.get("next_node") or "writer"
    if node in {"specialist", "writer", "hitl", "end"}:
        return node
    return "writer"


def _route_review(state: TeamState) -> str:
    if state.get("review_ok") or int(state.get("revisions") or 0) >= 2:
        return "end"
    return "writer"


def _build_graph(llm, agents: dict[str, FinanceReactAgent]):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        from langgraph.graph import END, StateGraph

        START = "__start__"

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", lambda state: supervisor_node(state, llm))
    graph.add_node("specialist", lambda state: specialist_node(state, agents))
    graph.add_node("writer", lambda state: writer_node(state, llm))
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("hitl", hitl_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {"specialist": "specialist", "writer": "writer", "hitl": "hitl", "end": END},
    )
    graph.add_edge("specialist", "supervisor")
    graph.add_edge("hitl", END)
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", _route_review, {"writer": "writer", "end": END})
    return graph.compile()


def _specialists_from(base: FinanceReactAgent) -> dict[str, FinanceReactAgent]:
    shared = {
        "client": base.client,
        "tools": base.tools,
        "llm": base.llm,
    }
    return {
        "spend": FinanceReactAgent(prompt=SPEND_PROMPT, allowed_tools=["spending_stat"], **shared),
        "notes": FinanceReactAgent(prompt=NOTES_PROMPT, allowed_tools=["search_notes"], **shared),
        "gold": FinanceReactAgent(prompt=GOLD_PROMPT, allowed_tools=["gold_price"], **shared),
        "email": FinanceReactAgent(
            prompt=EMAIL_PROMPT,
            allowed_tools=["spending_stat", "send_email"],
            **shared,
        ),
    }


class FinanceTeam:
    def __init__(self, researcher: FinanceReactAgent | None = None) -> None:
        self.researcher = researcher or FinanceReactAgent(prompt=RESEARCHER_PROMPT)
        self.agents = _specialists_from(self.researcher)
        self.graph = _build_graph(self.researcher.llm, self.agents)

    def tool_names(self) -> list[str]:
        names: list[str] = []
        for agent in self.agents.values():
            for name in agent.tool_names():
                if name not in names:
                    names.append(name)
        return names

    def ask(
        self,
        query: str,
        thread_id: str = "team-thread",
        *,
        hitl_decision: str | None = None,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        resume = resume or {}
        payload: dict[str, Any] = {
            "question": query,
            "thread_id": thread_id,
            "revisions": 0,
            "agent_trace": list(resume.get("agent_trace") or []),
            "review_issues": [],
            "completed_specialists": list(resume.get("completed_specialists") or []),
            "research_stats": dict(resume.get("research_stats") or {}),
            "rag_hits": list(resume.get("rag_hits") or []),
            "research_text": resume.get("research_text") or "",
            "react_trace": list(resume.get("react_trace") or []),
            "hitl_decision": hitl_decision or "",
            "supervisor_loops": 0,
            "metrics": dict(resume.get("metrics") or {"retries": 0, "specialist_calls": 0}),
        }
        log_event("run_start", thread_id=thread_id, agent="team", detail=query[:200], extra={"hitl": hitl_decision or ""})
        result = self.graph.invoke(payload)
        elapsed = int((time.perf_counter() - started) * 1000)
        metrics = dict(result.get("metrics") or {})
        metrics["elapsed_ms"] = elapsed
        structured = result.get("structured") or {
            "income": float((result.get("research_stats") or {}).get("income_total") or 0),
            "expense": float((result.get("research_stats") or {}).get("expense_total") or 0),
            "source_tool": (result.get("research_stats") or {}).get("source_tool") or "spending_stat",
        }
        try:
            structured = FinanceReport.model_validate(structured).model_dump()
        except Exception:
            pass
        pending = result.get("pending_hitl")
        stats = dict(result.get("research_stats") or payload.get("research_stats") or {})
        if (
            not pending
            and wants_email(query)
            and extract_email(query)
            and (hitl_decision or "").lower() not in {"approve", "reject"}
            and not stats.get("email_result")
        ):
            paused = hitl_node(
                {
                    "question": query,
                    "research_stats": stats,
                    "rag_hits": result.get("rag_hits") or [],
                    "research_text": result.get("research_text") or "",
                    "completed_specialists": result.get("completed_specialists") or [],
                    "agent_trace": result.get("agent_trace") or [],
                    "react_trace": result.get("react_trace") or [],
                    "metrics": metrics,
                    "thread_id": thread_id,
                }
            )
            pending = paused.get("pending_hitl")
            result["answer"] = paused.get("answer")
        log_event(
            "run_end",
            thread_id=thread_id,
            agent="team",
            detail=f"{elapsed}ms pending_hitl={bool(pending)}",
            extra=metrics,
        )
        return {
            "answer": result.get("answer") or result.get("draft") or "",
            "trace": result.get("react_trace") or [],
            "team": result.get("agent_trace") or [],
            "structured": structured,
            "rag_hits": result.get("rag_hits") or [],
            "review_ok": bool(result.get("review_ok")),
            "review_issues": result.get("review_issues") or [],
            "guardrails": [result.get("email_gate") or check_email_request(query)],
            "tools": self.tool_names(),
            "pending_hitl": pending,
            "metrics": metrics,
            "routing_via": result.get("routing_via"),
            "completed_specialists": result.get("completed_specialists") or [],
            "research_stats": stats,
        }
