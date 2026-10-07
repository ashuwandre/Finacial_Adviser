# Financial Adviser

Study project: a household-finance chatbot built as a **LangGraph ReAct agent** that discovers tools through **MCP**.

```
User
  -> Streamlit chat
      -> ReAct agent (Azure OpenAI + memory)
          -> MCP client
              -> MCP server tools
                  -> FastAPI /stats
                  -> SMTP email
                  -> gold price API
```

## What you can ask

- How much did I spend on food in 2018?
- What is my total income?
- Which category has the highest expenses?
- What is today's gold price?
- Email my spending summary to me@gmail.com

The UI shows the ReAct trace: Thought, Act, Observe, Answer.

## Setup

```powershell
cd "C:\Users\Ashwini Wandre\OneDrive\Desktop\Finacial_Adviser"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirement.txt
```

Fill `app/.env` using `app/.env.example`. Azure OpenAI is required for chat. SMTP is only required for email.

The backend loads `app/default_data.csv` if present, otherwise `pandas/Daily Household Transactions.csv`.

## Run

Terminal 1 — FastAPI:

```powershell
cd "C:\Users\Ashwini Wandre\OneDrive\Desktop\Finacial_Adviser"
.\.venv\Scripts\Activate.ps1
cd app
python backend_finacial_adviser.py
```

Terminal 2 — Streamlit. This process starts the MCP HTTP server on port 8001:

```powershell
cd "C:\Users\Ashwini Wandre\OneDrive\Desktop\Finacial_Adviser"
.\.venv\Scripts\Activate.ps1
streamlit run app\ui_streamlit.py
```

Open http://localhost:8501. Backend health: http://127.0.0.1:8000/health

## Project files

| File | Role |
|---|---|
| `app/ui_streamlit.py` | Chat UI, CSV upload, category HITL, ReAct traces |
| `app/react_agent.py` | ReAct agent, MCP client, memory, trace extraction |
| `app/mcp_server.py` | MCP tools: spending_stat, send_email, gold_price |
| `app/backend_finacial_adviser.py` | FastAPI stats over the CSV |
| `app/category_rules.json` | Category -> Income / Expense / Transfer |

## Agent concepts already in this build

- **LLM + system prompt:** Azure OpenAI with a finance role
- **Tool calling:** the model chooses a tool instead of inventing numbers
- **ReAct loop:** Thought -> Act -> Observe until it can answer
- **MCP:** tools are published by a server and discovered by the client
- **Short-term memory:** LangGraph checkpointer + `thread_id`
- **Human-in-the-loop:** you classify unknown categories
- **Observability:** the UI prints the ReAct and team traces
- **Multi-agent:** researcher -> writer -> reviewer
- **RAG:** TF-IDF search over transaction notes (`/search_notes`)
- **Guardrails:** refuse email with no address; block invented amounts
- **Structured output:** `{ "income", "expense", "source_tool" }`

Study business need: produce a short household money brief from real totals and notes.

## Files added for this study layer

| File | Role |
|---|---|
| `app/multi_agent.py` | Researcher / writer / reviewer graph |
| `app/rag_notes.py` | TF-IDF note retrieval |
| `app/guardrails.py` | Email + invented-amount rules and JSON schema |

## Later upgrades

- **Planning:** a planner step before tool calls for compare-year questions
- **Long-term memory:** save preferred email and currency
- **Reflection:** a second math-check pass
- **Streaming:** token-by-token answers
- **Evaluation:** gold questions and expected tool calls
