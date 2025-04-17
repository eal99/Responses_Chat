import streamlit as st
from openai import OpenAI
from streamlit_chat import message
import base64
import re

###############################################################################
# 1. Configuration & Session State
###############################################################################
st.set_page_config(
    page_title="Impacked Packaging Chat Search (GPT-4.1)",
    layout="centered",
    page_icon="📦"
)
client = OpenAI()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None
if "developer_instructions" not in st.session_state:
    st.session_state.developer_instructions = """
# Identity
You are an expert assistant specializing in packaging and cosmetics products. You help users find matching products by searching a vector store based on their queries.

# Instructions

- Listen carefully to the user's queries.
- Interpret ambiguous/broad queries using your expertise.
- If possible, conduct a thorough search in the vector store for matches.
- When needed, ask concise follow-up questions **before** searching again to clarify user needs.
- Never guess at search results; use your tools for reliable answers.
- Plan and reflect before searching; if previous searches were unhelpful, recap and try alternate approaches.

# Steps

1. **Understand the Query:** Analyze the initial user query for product info or categories.
2. **Search the Vector Store:** With sufficient info, use file_search in the Impacked Packaging vector store (`vs_67fd31e9c4c081919a9c34d1be81e2d9`).
3. **Follow-Up Questions:** If the query/result is unclear/broad, ask for clarification.
4. **Product Suggestions:** Suggest matching products concisely, showing up to 5 at a time with short descriptors and images (unless "Plastirey").
5. **Further Assistance:** Offer additional guidance, or ask if the user wants another search.

# Examples

**User:** I'm looking for eco-friendly packaging options for skincare products.
**Assistant:**
- You’re looking for eco-friendly packaging for skincare.
- Initial options include:
  - CKS Packaging 2oz Cosmo Round – HDPE, 2oz, portable, retail bottle.
    ![Product image](https://generic.webpackaging.com/img/live/2537/13914521/13550373/13570224-DWIKMYMB/main/Mold_469.png)
Are you looking for bottles, tubes, or jars? Any specific material (glass, PCR, etc.)? Reply to narrow options!

# Notes

- Surface follow-up questions concisely.
- Consider user preferences like material, sustainability, or product type.
- Stay up-to-date with packaging trends and products.
- Do not display product images for Plastirey products.
    """

###############################################################################
# 2. Helper Functions
###############################################################################
def construct_input_messages(history, user_input, image_url=None, developer_instructions=None):
    input_list = []
    if developer_instructions:
        input_list.append({"role": "developer", "content": developer_instructions})
    for msg in history:
        input_list.append({"role": msg["role"], "content": msg["content"]})
    # The newest user input
    if image_url:
        input_list.append({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": user_input}
            ]
        })
    else:
        input_list.append({"role": "user", "content": user_input})
    return input_list

def display_message_bubble(msg, idx):
    # User/assistant message bubble using streamlit_chat
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{idx}", avatar_style='mini')
    else:
        content = msg["content"]
        urls = re.findall(r"image_url\s*:\s*(https?://\S+)", content)
        cleaned_content = re.sub(r"image_url\s*:\s*https?://\S+", "", content).strip()
        message(cleaned_content, key=f"assistant_{idx}")
        # Show images if not Plastirey
        for url in urls:
            if "plastirey" not in content.lower():
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin: -0.5em 0 1.3em 2.5em;">
                        <img src="{url}" style="max-width:180px; border-radius:0.5em; box-shadow:0 2px 6px #0002;" />
                    </div>
                    """, unsafe_allow_html=True
                )

###############################################################################
# 3. UI Construction & Main App
###############################################################################
hide_streamlit_style = """
<style>
[data-testid="stSidebar"] {display: none;}
.block-container {padding-top: 1.5rem;}
#MainMenu, footer {visibility: hidden;}
.stChatInput {bottom: 0 !important;}
.stTextInput > div > input { font-size: 1.15rem;}
.chat-history {min-height: 500px; max-height:72vh; overflow-y: auto;}
.user-bubble {
    background: #2e2e38 !important;
    color: #fff !important;
    border-radius: 10px 10px 2px 10px;
    margin-bottom: 0.5em;
}
.assistant-bubble {
    background: #ededed !important;
    border-radius: 10px 10px 10px 2px;
    margin-bottom: 0.5em;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("📦 Impacked Packaging Search – Chat UI (GPT-4.1)")
st.caption("Ask product questions or try: 'Show me eco-friendly 100ml airless bottles for skincare'.")

# Developer instructions editor
with st.expander("⚙️ Developer Instructions (Advanced)", expanded=False):
    updated_instructions = st.text_area(
        "Instructions:",
        value=st.session_state.developer_instructions,
        height=250
    )
    st.session_state.developer_instructions = updated_instructions

# Image uploader (optional)
with st.expander("Optional: Upload an image (for visual search, e.g. packaging ref)", expanded=False):
    uploaded_img = st.file_uploader("Upload image (png, jpg)", type=["png", "jpg", "jpeg"])
    image_url = None
    if uploaded_img:
        image_bytes = uploaded_img.read()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_img.type or "image/png"
        image_url = f"data:{mime_type};base64,{b64_data}"
        st.image(image_bytes, caption="Your uploaded image", use_column_width=True)

# Display chat history/bubbles
st.markdown("""<div class='chat-history'>""", unsafe_allow_html=True)
for idx, msg in enumerate(st.session_state.messages):
    display_message_bubble(msg, idx)
st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input (ChatGPT-style): supports Enter-to-send and button
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your question...", "",
        key="chat_input",
        placeholder="Ask about eco-friendly bottles, compare materials, etc."
    )
    submitted = st.form_submit_button("Send 🚀")

if submitted and user_input.strip():
    # Add user's message to session
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    input_messages = construct_input_messages(
        st.session_state.messages[:-1],
        user_input.strip(),
        image_url=image_url,
        developer_instructions=st.session_state.developer_instructions
    )
    # Stream response from GPT-4.1
    partial_assistant_text = ""
    try:
        with st.spinner("Impacked GPT-4.1 is thinking..."):
            stream = client.responses.create(
                model="gpt-4.1",
                instructions=None,  # Instructions already in 'developer' role in inputs
                input=input_messages,
                stream=True,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["vs_67fd31e9c4c081919a9c34d1be81e2d9"]
                }],
                temperature=0.8,
                max_output_tokens=4096,
                top_p=1,
                store=True
            )
            stream_box = st.empty()
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta" and hasattr(event, "delta"):
                    partial_assistant_text += event.delta
                    stream_box.markdown(f"**Assistant:** {partial_assistant_text}")
            # Finalize after stream
            st.session_state.messages.append({"role": "assistant", "content": partial_assistant_text})
    except Exception as e:
        st.error(f"Error with GPT-4.1: {e}")

# Optional: clear message history
with st.expander("🧹 Clear chat history", expanded=False):
    if st.button("Reset Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.rerun()

###############################################################################
# 4. Suggestions & UX Enhancements
###############################################################################
st.markdown(
    """
    <hr>
    <sup>
    <b>Tips:</b>
    - Paste product images to the chat to trigger visual search.<br>
    - Ask for comparisons ("Compare PET vs. glass bottles for serums") or sustainability details.<br>
    <b>Demo by Impacked Packaging · Powered by OpenAI GPT-4.1</b>
    </sup>
    """, unsafe_allow_html=True
)