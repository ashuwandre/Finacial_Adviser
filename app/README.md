# 💰 Financial Advisor AI

An AI-powered financial advisor built using **Streamlit**, **LangGraph**, **LangChain**, and **Azure OpenAI**. The application allows users to analyze personal financial transactions, ask questions in natural language, upload new transaction files, receive spending insights, check live gold prices, and send financial reports via email.

---

# Features

### 📊 Spending Analysis

* Analyze income and expenses.
* View spending statistics by:

  * Year
  * Month
  * Category
  * Income/Expense direction
* Calculate total income and expenses.
* Generate financial summaries.

### 🤖 AI Financial Assistant

Powered by **Azure OpenAI** and **LangGraph**.

Users can ask questions such as:

* How much did I spend in 2024?
* Show my food expenses.
* What is my total income this year?
* Which category has the highest spending?
* Summarize my financial data.

The AI automatically calls backend tools to retrieve the required information.

---

### 📁 CSV Upload

Upload your bank transaction CSV file directly from the UI.

After uploading:

* CSV is stored in the uploads folder.
* Backend reloads the data.
* Financial statistics are refreshed automatically.

---

### 🏷 Unknown Category Classification

If new transaction categories are detected:

* The application displays them.
* Users can classify them as:

  * Expense
  * Income
  * Transfer

The rules are saved for future processing.

---

### 📧 Email Reports

The assistant can send financial summaries via email using SMTP.

Example prompt:

> Email my spending summary to [example@gmail.com](mailto:example@gmail.com)

---

### 🪙 Live Gold Price

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

```
User
   │
   ▼
Streamlit UI
   │
   ▼
LangGraph Agent
   │
   ├──────── Spending Tool
   │
   ├──────── Email Tool
   │
   └──────── Gold Price Tool
   │
   ▼
FastAPI Backend
   │
   ▼
CSV Transaction Data
```
 

 
Technologies: Python, Machine Learning, LangChain, LangGraph, Azure OpenAI, FastAPI,  