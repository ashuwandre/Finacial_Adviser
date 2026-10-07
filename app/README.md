<<<<<<< HEAD
# Financial Adviser

Study project: a household-finance chatbot built as a **LangGraph ReAct agent** that discovers tools through **MCP**.
=======
# Financial Advisor AI

An AI-powered financial advisor built using **Streamlit**, **LangGraph**, **LangChain**, and **Azure OpenAI**. The application allows users to analyze personal financial transactions, ask questions in natural language, upload new transaction files, receive spending insights, check live gold prices, and send financial reports via email.

---

# Features

### Spending Analysis

* Analyze income and expenses.
* View spending statistics by:

  * Year
  * Month
  * Category
  * Income/Expense direction
* Calculate total income and expenses.
* Generate financial summaries.

### AI Financial Assistant

Powered by **Azure OpenAI** and **LangGraph**.

Users can ask questions such as:

* How much did I spend in 2024?
* Show my food expenses.
* What is my total income this year?
* Which category has the highest spending?
* Summarize my financial data.

The AI automatically calls backend tools to retrieve the required information.

---

### CSV Upload

Upload your bank transaction CSV file directly from the UI.

After uploading:

* CSV is stored in the uploads folder.
* Backend reloads the data.
* Financial statistics are refreshed automatically.

---

### Unknown Category Classification

If new transaction categories are detected:

* The application displays them.
* Users can classify them as:

  * Expense
  * Income
  * Transfer

The rules are saved for future processing.

---

### Email Reports

The assistant can send financial summaries via email using SMTP.

Example prompt:

> Email my spending summary to [example@gmail.com](mailto:example@gmail.com)

---

### Live Gold Price

Fetches the latest gold spot price using the Metals Live API.

Example prompt:

> What is today's gold price?

---

## Tech Stack

| Component              | Technology        |
| ---------------------- | ----------------- |
| Frontend               | Streamlit         |
| Backend                | FastAPI           |
| LLM Framework          | LangChain         |
| Agent Framework        | LangGraph         |
| AI Model               | Azure OpenAI      |
| Data Processing        | Pandas            |
| HTTP Client            | Requests          |
| Environment Management | python-dotenv     |
| Email Service          | SMTP              |
| File Storage           | Local File System |

---

# Project Structure

```
FinancialAdvisor/
│
├── app/
│   ├── main.py
│   ├── .env
│   └── ...
│
├── uploads/
│
├── backend/
│   ├── FastAPI APIs
│   ├── CSV Processing
│   └── Statistics Engine
│
├── .env
├── requirements.txt
└── README.md
```

---

# AI Tools

The LangGraph agent has access to the following tools:

## Spending Statistics Tool

Returns spending statistics from the backend.

Parameters:

* year
* month
* category
* direction

Backend endpoint:

```
GET /stats
```

---

## Email Tool

Sends financial reports using SMTP.

Parameters:

* recipient
* subject
* body

---

## Gold Price Tool

Fetches the latest live gold price.

API:

```
https://api.metals.live/v1/spot/gold
```

---

# Backend APIs

| Method | Endpoint | Description                        |
| ------ | -------- | ---------------------------------- |
| GET    | /stats   | Get spending statistics            |
| GET    | /filters | Get available years and categories |
| GET    | /reload  | Reload CSV data                    |
| POST   | /set_csv | Upload new CSV                     |
| POST   | /rules   | Save category classification rules |

---

# Environment Variables

Create a `.env` file.

```
AZURE_OPENAI_DEPLOYMENT_NAME=

AZURE_OPENAI_MODEL_NAME=

AZURE_OPENAI_API_VERSION=

AZURE_OPENAI_API_BASE=

AZURE_OPENAI_API_KEY=

BACKEND_URL=http://localhost:8000

SMTP_SERVER=

SMTP_PORT=587

SMTP_USER=

SMTP_PASSWORD=
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Start the FastAPI backend.

```bash
uvicorn backend.main:app --reload
```

---

# Running the Streamlit App

```bash
streamlit run app/main.py
```

---

# Example Questions

* How much did I spend in 2023?
* Show my grocery expenses.
* What is my total income?
* Compare spending between 2023 and 2024.
* Which category has the highest expenses?
* Email my financial summary.
* What is today's gold price?
* Show expenses for March 2024.
* Give me a spending summary.
* Which month had the highest expenses?

---

# Workflow
>>>>>>> 38238a14d2ee90800688b0f4ab3198001f9dabe5

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

<<<<<<< HEAD
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
=======
 
Technologies: Python, Machine Learning, LangChain, LangGraph, Azure OpenAI, FastAPI,  
>>>>>>> 38238a14d2ee90800688b0f4ab3198001f9dabe5
