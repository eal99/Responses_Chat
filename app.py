"""
PackagingPal – AI Sourcing Assistant (v3)
A refreshed Streamlit UI + cleaner instruction set
Author: ChatGPT ("Sam")
Date: 2025‑04‑17
"""
# ---------------------------------------------------------------------------
# 0) Imports & basic config
# ---------------------------------------------------------------------------
import json, base64, requests, textwrap, datetime
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI

# -- visual & page meta ------------------------------------------------------
st.set_page_config(
    page_title="PackagingPal – AI Sourcing Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject a tiny bit of CSS to beautify the chat bubbles & product cards -------
st.markdown(
    """
    <style>
    .chat-bubble-user        {background:#DCF8C6;padding:12px 16px;border-radius:12px;margin:4px 0;}
    .chat-bubble-assistant   {background:#F1F0F0;padding:12px 16px;border-radius:12px;margin:4px 0;}
    img.product-thumb        {border-radius:8px;width:100%;object-fit:cover;box-shadow:0 2px 6px rgba(0,0,0,.08);}    
    .card                    {border:1px solid #e2e2e2;border-radius:12px;padding:16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1) Constants – API endpoints, System Prompt, etc.
# ---------------------------------------------------------------------------
API_BASE_URL     = "https://freeform-search-impacked-19a53d14c347.herokuapp.com/api"
VECTOR_STORE_ID  = st.secrets["VECTOR_ID"]
MODEL_NAME       = "gpt-4.1"
MAX_TOKENS_RESP  = 1200
OPENAI_API_KEY   = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are **PackagingPal**, the expert sourcing assistant for sustainable cosmetic & personal‑care packaging.

    ## Modes
    1. **CLARIFY** – Ask short follow‑up questions until requirements (volume, material, MOQ, look & feel) are clear.
    2. **SEARCH**  – Once enough detail is gathered, output **only** the line:
       ```
       ###SEARCH### <query>
       ```
       where `<query>` is a concise search string (≤ 20 words, no quotes).
    3. **RECOMMEND** – When product hits are supplied (both from Qdrant & File‑Search), analyse them **holistically** and reply with:
       * Top 5 SKU recommendations in a tidy Markdown table
       * A short rationale (≤ 120 words)
       * Clear next steps / follow‑up questions if options are insufficient

    ## Output Rules
    * Never mention internal implementation details (vector store IDs, APIs, etc.).
    * All code/data blocks must be fenced with triple back‑ticks and a language hint (e.g. ```json).
    * Images must be rendered with standard Markdown: `![alt](url)`.
    * Remain professional, succinct, and objective.
    """
)

# ---------------------------------------------------------------------------
# 2) Streamlit session state helpers
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])      # chat transcript (role/content)
ss.setdefault("thread_id", None)   # OAI thread for the Responses API
ss.setdefault("ref_image", None)   # base64 data‑URI of uploaded image
ss.setdefault("debug", False)      # toggle debug prints

# ---------------------------------------------------------------------------
# 3) Utility functions
# ---------------------------------------------------------------------------

def impacked_search(query: str, k: int = 50) -> List[Dict]:
    """Hit Ethan's Flask /search endpoint (Qdrant hybrid)."""
    url = f"{API_BASE_URL}/search"
    payload = {"query": query, "top_k": k}
    if ss.debug:
        st.write("[DEBUG] POST", url, payload)
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"❌ Impacked /search error: {exc}")
        return []


def first_image(p: Dict) -> Optional[str]:
    """Return the first non‑empty image URL from product record."""
    for key in [*(f"image_url_{i}" for i in range(1, 6)), "image_url", "image"]:
        if p.get(key):
            return p[key]
    imgs = p.get("image_urls")
    if isinstance(imgs, list) and imgs:
        return str(imgs[0]).strip("{} ")
    return None


def hits_for_llm(hits: List[Dict], k: int = 12) -> str:
    """Trim & normalise for the LLM."""
    return json.dumps(
        [
            {
                "id": x.get("id"),
                "title": x.get("title"),
                "supplier": x.get("company_name"),
                "score": round(x.get("score", 0), 3),
                "image_url": first_image(x),
            }
            for x in hits[:k]
        ],
        ensure_ascii=False,
    )


def send(events, tools=None, max_tokens=MAX_TOKENS_RESP):
    """Wrapper around client.responses.create with streaming."""
    params = dict(model=MODEL_NAME, input=events, stream=True, max_output_tokens=max_tokens)
    if tools:
        params["tools"] = tools
    if ss.thread_id:
        params["thread_id"] = ss.thread_id

    collected = ""
    for chunk in client.responses.create(**params):
        # capture thread once available
        if not ss.thread_id and getattr(chunk, "thread_id", None):
            ss.thread_id = chunk.thread_id
        if chunk.type == "response.output_text.delta" and hasattr(chunk, "delta"):
            collected += chunk.delta
            yield chunk.delta  # stream back to UI
    return collected

# ---------------------------------------------------------------------------
# 4) Sidebar – settings & controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️  Controls")
    if st.button("🔄 New conversation"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    ss.debug = st.toggle("Show debug logs", value=ss.debug)
    with st.expander("System prompt", expanded=False):
        ss.system_prompt = st.text_area("Edit SYSTEM_PROMPT if needed", value=ss.get("system_prompt", SYSTEM_PROMPT), height=350)



# ---------------------------------------------------------------------------
# 5) Optional reference image upload
# ---------------------------------------------------------------------------
up = st.file_uploader("📸 Upload reference image (optional)", ["png", "jpg", "jpeg"])
if up:
    mime = up.type or "image/png"
    ss.ref_image = f"data:{mime};base64,{base64.b64encode(up.read()).decode()}"
    st.image(ss.ref_image, caption="Reference image", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 6) Chat Input (bottom‑anchored)
# ---------------------------------------------------------------------------
user_query = st.chat_input("Type your message…")  # ⬅ automatically sticks to bottom

# ---------------------------------------------------------------------------
# 7) Conversation loop – process a new user message
# ---------------------------------------------------------------------------
if user_query:
    ss.messages.append({"role": "user", "content": user_query})

    # Build event list for GPT
    events = [{"role": "system", "content": ss.system_prompt}]
    events.extend(ss.messages[-20:])  # last 20 turns is plenty

    if ss.ref_image:
        events.append({"role": "user", "content": [{"type": "input_image", "image_url": ss.ref_image}]})

    # Stream assistant response ------------------------------------------------
    assistant_md = ""
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        for delta in send(events):
            assistant_md += delta
            placeholder.markdown(assistant_md)

    ss.messages.append({"role": "assistant", "content": assistant_md})

    # -----------------------------------------------------------------------
    # If the assistant triggered a SEARCH phase, run backend query & follow‑up
    # -----------------------------------------------------------------------
    if "###SEARCH###" in assistant_md:
        search_line = assistant_md.split("###SEARCH###", 1)[1].strip().splitlines()[0]
        if ss.debug:
            st.write("[DEBUG] Triggered SEARCH →", search_line)

        hits = impacked_search(search_line, k=50)
        if ss.debug:
            st.write("[DEBUG] Hits", len(hits))

        # Display results as a responsive grid (UI candy) -------------------
        if hits:
            st.subheader("🔍 Candidate products")
            cols = st.columns(4)
            for idx, h in enumerate(hits[:8]):  # top 8 thumbnails
                with cols[idx % 4]:
                    img = first_image(h)
                    if img:
                        st.image(img, caption=h.get("title", "(no title)"), use_container_width=True, clamp=True)

        # Prepare recommendation prompt ------------------------------------
        rec_events = [
            {"role": "system", "content": ss.system_prompt},
            {"role": "user", "content": f"IMPACKED_HITS:\n```json\n{hits_for_llm(hits)}\n```"},
        ]
        if ss.ref_image:
            rec_events.append({"role": "user", "content": [{"type": "input_image", "image_url": ss.ref_image}]})

        tools = [{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}]

        rec_md = ""
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for delta in send(rec_events, tools=tools, max_tokens=1800):
                rec_md += delta
                placeholder.markdown(rec_md)

        ss.messages.append({"role": "assistant", "content": rec_md})

    st.rerun()  # show the fresh chat history with the new turn

# ---------------------------------------------------------------------------
# 8) Render existing chat history -------------------------------------------
# ---------------------------------------------------------------------------
for m in ss.messages:
    role = m["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"], unsafe_allow_html=True)

st.caption(f"© {datetime.datetime.now().year} PackagingPal – streamlit prototype")
