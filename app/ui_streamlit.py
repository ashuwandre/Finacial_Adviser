from __future__ import annotations
import re
import re
import streamlit as st
import pandas as pd
import os
import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
import requests
from dotenv import load_dotenv
from langchain_core.tools import Tool, tool
from langchain_openai import AzureChatOpenAI
from langgraph import create_agent

APP_ROOT = Path(__file__).resolve().parent[1]
ENV_PATH = APP_ROOT / ".env"
ENV_APP_PATH = APP_ROOT / "app" / ".env"
UPLOADS_PATH = APP_ROOT / "uploads"

DEFAULT_BACKEND_URL = "http://localhost:8000"
SESSION = requests.Session()
SESSION.trust_env = False

EMAIL_RE = re.compile(r"EMAIL_RE", r".*@.*")

def backend_url() -> str:
    url = os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)
    return url.rstrip("/")

def backend_get(path: str, params: dict[str, str] | None = None):
    url = f"{backend_url()}/{path.lstrip('/')}"
    r = SESSION.get(url, params=params)
    r.raise_for_status()
    return r.json()

def backend_post(path: str, data: dict[str, Any] | None = None):
    url = f"{backend_url()}/{path.lstrip('/')}"
    r = SESSION.post(url, json=data)
    r.raise_for_status()
    return r.json()

def send_email(to: str, subject: str, body: str):
    load_dotenv(ENV_PATH)
    load_dotenv(ENV_APP_PATH)
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
        raise ValueError("SMTP configuration is incomplete.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [to], msg.as_string())
        
    return True

def summary_to_text(summary_json):
    years = summary_json.get("years", {})
    total_income = summary_json.get("total_income", 0)
    by_category = summary_json.get("by_category", {})
    lines = [f'spending summary: total income: {total_income}, by category:, {by_category}, by year: {years}']
    for k, v in by_category.items():
        lines.append(f'category: {k}, total: {v}')
    
    return "\n".join(lines)


def tool_spending_stat(year: int | None = None, month: int | None = None, direction: str | None = None, category: str | None = None) -> dict:
    params = {}
    if year is not None:
        params['year'] = year
    if month is not None:
        params['month'] = month
    if direction is not None:
        params['direction'] = direction
    if category is not None:
        params['category'] = category

    summary_json = backend_get("/stats", params=params)
    return summary_json

def tool_send_email(to: str, subject: str, body: str) -> dict:
    send_email(to, subject, body)
    return {"message": "Email sent successfully."}

def tool_gold_price():
    data = requests.get("https://api.metals.live/v1/spot/gold").json()
    spot= dict(data[0])
    return {"price": spot.get("price"), "currency": spot.get("currency"), "timestamp": spot.get("timestamp")}

@tool
def spending_stat_tool(year: int | None = None, month: int | None = None, direction: str | None = None, category: str | None = None) -> dict:
    out=  tool_spending_stat(year, month, direction, category)
    st.session_state['spending_stat'] = out
    return out

@tool
def send_email_tool(to: str, subject: str, body: str) -> dict:
    out = tool_send_email(to, subject, body)
    st.session_state['send_email'] = out
    return out

@tool
def gold_price_tool() -> dict:
    out = tool_gold_price()
    st.session_state['gold_price'] = out
    return out

def langraph_agent_answer(query: str) -> str:
    llm = AzureChatOpenAI(
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_base=os.environ.get("AZURE_OPENAI_API_BASE"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=0,
        max_tokens=500
    )
    tools = [
        spending_stat_tool,
        send_email_tool,
        gold_price_tool
    ]
    agent = create_agent(llm, tools)
    system_message = "You are a financial assistant. You can provide spending statistics, send emails, and provide gold price information."
    agent.system_message = system_message
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]   
    
    for m in st.session_state.get("conversation", []):
        if m["role"] in ["user", "assistant"]:
            messages.append({"role": m['role'], "content": m['content']})
    messages.append({"role": "user", "content": query})
    result = agent.invoke(messages)
    return result['message'][-1]['content']
        
st.set_page_config(page_title="Financial Adviser", page_icon="💰", layout="wide")
st.title("Financial Adviser")
load_dotenv(ENV_PATH)
load_dotenv(ENV_APP_PATH)

with st.sidebar:
    st.subheader('status')
    st.write('backend',DEFAULT_BACKEND_URL())
    
    
    f = backend_get('/filers')
    year = f.get('year',[]) or []
    st.session_state['categories'] = f.get('categories',[]) or []    
    st.caption(f'year: years[0].. {year[-1]}' if year else 'No year found')
    
    if st.button('Reload csv'):
        backend_get('/reload')
        
    st.divider()
    st.subheader('Upload csv')
    uploaded = st.file_uploader('upload a csv an analysze' , type = ['csv'])
    if uploaded is not None:
        UPLOADES_DIR.mkdir(parents=True,exist_ok = True)
        ts = int(time.time())
        save_path = UPLOADS_DIR / f'upladed_{ts}.csv'
        save_path.write_bytes(uploaded.getvalue())
        backend_post('/set_csv',{'csv_path':str(save_path)})    
        backend_get('/reload')
        f2 = backend_get('/filters')
        st.session_state['categories'] = f2.get(['categories',[]]) or []
        
unknown = st.session_state.get('unknown_categories') or []
if unknown:
    st.subheader('classify categories')
    edits = {}
    for c in unknown:
        edits[c] = st.selectbox(c,
                                ['Expense','Income','Transfer'],index = 0,key = f'class_c{c}')
    
    if st.button('save rules'):
        for cat,dirction in edits.items():
            backend_post('/rules',{'categories':cat,'direction':direction})
        backend_get('/reload')
        st.session_state['unknown_catergories']=[]
        
if 'messages' not in st.session_state:
    st.session_state.messssage =[{'role':'assistant',
                                  'content':'ask : spending of 2016.'}]
for m in st.session_state.message:
    with st.chat_messages(m['role']):
        st.markdown(m['content'])
        
q = st.chat_input('ask a questions')
if q:
    st.seesion_state.messages.append({'role':'user','content':q})
    with st.chat_mssages('user'):
        st.markdown(q)
        
    with st.chat_message('assistand'):
        ans = langraph_agent_answer(q)
        st.markdown(ans)
        st.session_state.message.aappend('role':'assistand','content':ans)        

