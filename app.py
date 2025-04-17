"""
PackagingPal – AI Sourcing Assistant (v2  stable)
"""
# ---- imports & config -------------------------------------------------------
import json, base64, requests
from typing import List, Dict, Optional
import streamlit as st
from openai import OpenAI

st.set_page_config("PackagingPal – AI Sourcing Assistant", layout="wide")

API_BASE_URL   = "https://freeform-search-impacked-19a53d14c347.herokuapp.com/api"
VECTOR_STORE_ID = "vs_67fd31e9c4c081919a9c34d1be81e2d9"
client = OpenAI()

# ---- session state ----------------------------------------------------------
ss = st.session_state
if "messages" not in ss:  ss.messages = []
if "thread_id" not in ss: ss.thread_id = None
if "image"     not in ss: ss.image     = None

SYSTEM_PROMPT = """
You are **PackagingPal** …
(unchanged text with ###SEARCH### and JSON‑block instructions)
"""

# ---- helpers ---------------------------------------------------------------
def impacked_search(q: str, k: int = 50) -> List[Dict]:
    url = f"{API_BASE_URL}/search"
    payload = {"query": q, "top_k": k}
    print(f"[DEBUG] POST {url} | {payload}", flush=True)
    try:
        r = requests.post(url, json=payload, timeout=25)
        print(f"[DEBUG] status {r.status_code}", flush=True)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Impacked /search error: {exc}")
        return []

def first_image(p: Dict) -> Optional[str]:
    for k in ("image_url", *(f"image_url_{i}" for i in range(1, 6))):
        if p.get(k): return p[k]
    imgs = p.get("image_urls")
    if isinstance(imgs, list) and imgs: return str(imgs[0]).strip("{} ")
    return None

def hits_for_llm(h: List[Dict], k: int = 12) -> str:
    return json.dumps([{
        "id":     x.get("id"),
        "title":  x.get("title"),
        "supplier": x.get("company_name"),
        "score":  round(x.get("score", 0), 3),
        "image_url": first_image(x)
    } for x in h[:k]], ensure_ascii=False)

def send(events, tools=None, max_tokens=1200):
    print(f"[DEBUG] send | tools={tools}", flush=True)
    params = dict(model="gpt-4.1", input=events, stream=True,
                  max_output_tokens=max_tokens)
    if tools:        params["tools"] = tools
    if ss.thread_id: params["thread_id"] = ss.thread_id
    txt = ""
    for ev in client.responses.create(**params):
        if not ss.thread_id and getattr(ev, "thread_id", None):
            ss.thread_id = ev.thread_id
        if ev.type == "response.output_text.delta" and hasattr(ev, "delta"):
            txt += ev.delta
    print(f"[DEBUG] send finished | chars={len(txt)}", flush=True)
    return txt

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    if st.button("🔄 New conversation"): ss.clear(); st.rerun()
    ss.system = st.text_area("System prompt", value=ss.get("system", SYSTEM_PROMPT),
                             height=280)

# ---- optional image --------------------------------------------------------
up = st.file_uploader("Reference image (optional)", ["png","jpg","jpeg"])
if up:
    mime = up.type or "image/png"
    ss.image = f"data:{mime};base64,{base64.b64encode(up.read()).decode()}"
    st.image(ss.image, caption="Uploaded reference", use_container_width=True)

st.divider()

# ---- chat input ------------------------------------------------------------
query = st.text_input("Type your message", key="chat_input")
send_click = st.button("Send", use_container_width=True)

# ---- conversation logic ----------------------------------------------------
if send_click and query.strip():
    ss.messages.append({"role":"user","content":query})

    # ----------- build event list for GPT -----------------------------------
    events = [{"role":"system","content":ss.system},
              *[{k:v for k,v in m.items() if k in ("role","content")}
                for m in ss.messages[-20:]]]
    if ss.image:
        events.append({"role":"user",
                       "content":[{"type":"input_image","image_url": ss.image}]})

    with st.spinner("PackagingPal is thinking…"):
        assistant_text = send(events)

    ss.messages.append({"role":"assistant","content":assistant_text})

    # ----------- reached trigger? ------------------------------------------
    if "###SEARCH###" in assistant_text:
        search_q = assistant_text.split("###SEARCH###",1)[1].strip().splitlines()[0]
        print(f"[DEBUG] Trigger → {search_q}", flush=True)

        hits = impacked_search(search_q, k=50)
        print(f"[DEBUG] Impacked hits: {len(hits)}", flush=True)

        rec_events = [
            {"role":"system","content":ss.system},
            {"role":"user",
             "content": f"IMPACKED_HITS:\n```json\n{hits_for_llm(hits)}\n```"}]
        if ss.image:
            rec_events.append({"role":"user",
                               "content":[{"type":"input_image","image_url": ss.image}]})

        tools = [{"type":"file_search","vector_store_ids":[VECTOR_STORE_ID]}]

        with st.spinner("Retrieving products…"):
            rec_text = send(rec_events, tools=tools, max_tokens=1800)

        ss.messages.append({"role":"assistant","content":rec_text})

    st.rerun()   # refresh UI with the newly appended message(s)

# ---- render chat history ---------------------------------------------------
for m in ss.messages:
    st.chat_message(m["role"]).markdown(m["content"], unsafe_allow_html=True)

st.divider()