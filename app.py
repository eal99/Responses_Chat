import streamlit as st
from openai import OpenAI
from io import BytesIO
import base64
import re
from streamlit_chat import message  # Using streamlit-chat to render chat bubbles

###############################################################################
# 1) Configuration & Session State
###############################################################################

client = OpenAI()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "developer_instructions" not in st.session_state:
    st.session_state.developer_instructions = """
    You are an expert assistant specializing in packaging and cosmetics products. Your role is to help users find matching products by searching a vector store based on their query. Engage intuitively and provide assistance by asking follow-up questions when necessary (prior to executing another search of the files) to refine and narrow down the options available.

    When interacting with the user:

    - Listen carefully to the user's initial query.
    - Utilize your expertise to understand and interpret ambiguous or broad queries.
    - Conduct a thorough search in the vector store to identify potential matches.
    - If needed, formulate concise follow-up questions to gather additional information, helping to clarify user needs and refine search criteria.

    # Steps

    1. **Understand the Query:** Analyze the user's initial query for specific keywords or product categories.
    2. **Search the Vector Store:** Use the information provided to search the vector store for suitable product matches.
    3. **Follow-Up Questions:** If results are too broad or unclear, ask pertinent, concise questions to narrow down.
    4. **Provide Product Suggestions:** Offer a list of product matches based on refined criteria.
    5. **Further Assistance:** Be prepared to offer additional insights or suggestions if the user requires further information.

    # Output Format

    - Initial understanding in short bullet points summarizing the user query.
    - Concise follow-up questions if necessary, in full sentences.
    - A list of suggested products with brief descriptions.

    # Examples

    **Example 1:**

    **User Query:** "I'm looking for eco-friendly packaging options for skincare products."

    **Assistant Response:**
    Here are some initial suggestions:
    1. CKS Packaging 2oz Cosmo Round – 469

    Capacity: 2 oz  
    Material: HDPE, Round shape  
    Finish: 20mm-410, 20mm-415  
    Lightweight and portable—great for retail face moisturizer packaging  
    image_url:https://generic.webpackaging.com/img/live/2537/13914521/13550373/13570224-DWIKMYMB/main/Mold_469.png

    Are you interested in a specific type of skincare product packaging, such as bottles, tubes, or jars? Let me know if you have any specific requirements or if you'd like to see more products!

    # Notes

    - Consider user preferences like material choice, sustainability, and specific product types when suggesting options.
    - Stay up-to-date with the latest trends and products in packaging and cosmetics.
    - Display the images when possible. Do not display the images for Plastirey products.
    """

###############################################################################
# 2) Streamlit UI Layout
###############################################################################
st.set_page_config(page_title="GPT-4.1 Responses API Demo", layout="centered")
st.title("GPT-4.1 Chat (Responses API)")

# 2A) Developer instructions editor
st.markdown("#### Developer Instructions")
with st.expander("Edit developer instructions (advanced)", expanded=False):
    updated_instructions = st.text_area(
        "Edit developer instructions:",
        value=st.session_state.developer_instructions,
        height=200
    )
    st.session_state.developer_instructions = updated_instructions

# 2B) Optional Image uploader
st.markdown("#### Optional: Upload an image to pass to GPT-4.1")
uploaded_img = st.file_uploader("Upload an image (png, jpg, etc.)", type=["png", "jpg", "jpeg"])
image_url = None
if uploaded_img:
    image_bytes = uploaded_img.read()
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_img.type or "image/png"
    image_url = f"data:{mime_type};base64,{b64_data}"
    st.image(image_bytes, caption="Preview of your uploaded image", use_column_width=True)

st.markdown("---")
st.markdown("### Conversation History")

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        content = msg["content"]
        pattern = r"image_url\s*:\s*(https?://\S+)"
        urls = re.findall(pattern, content)
        # Display assistant message without image_url tag for cleanliness
        cleaned_content = re.sub(r"image_url\s*:\s*https?://\S+", "", content).strip()
        message(cleaned_content, key=f"assistant_{i}")
        if urls:
            for url in urls:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin: -0.5em 0 1.5em 2.5em;">
                        <img src="{url}" style="max-width: 220px; max-height: 160px; border-radius: 0.5em; box-shadow: 0 2px 6px rgba(0,0,0,0.16);" alt="Product image"/>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("#### Ask GPT-4.1 a question, or say something:")
user_input = st.text_input("Your message", "", key="chat_input")
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter some text before sending.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        input_messages = []
        developer_instructions = st.session_state.developer_instructions
        for msg in st.session_state.messages:
            input_messages.append({"role": msg["role"], "content": msg["content"]})
        if image_url:
            input_messages.append({
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url}
                ]
            })
        try:
            with st.spinner("GPT-4.1 is thinking..."):
                # If you want to keep the assistant stream expander, it only shows partials.
                with st.expander("Assistant Stream (click to expand)", expanded=False):
                    partial_display = st.empty()
                    partial_assistant_text = ""
                    stream = client.responses.create(
                        model="gpt-4.1",
                        instructions=developer_instructions,
                        input=input_messages,
                        stream=True,
                        reasoning={},
                        tools=[{
                            "type": "file_search",
                            "vector_store_ids": ["vs_67fd31e9c4c081919a9c34d1be81e2d9"]
                        }],
                        temperature=1,
                        max_output_tokens=16384,
                        top_p=1,
                        store=True
                    )
                    for event in stream:
                        if event.type == "response.output_text.delta" and hasattr(event, "delta"):
                            partial_assistant_text += event.delta
                            partial_display.markdown(f"**Assistant**: {partial_assistant_text}")
                # After streaming, append final message
                st.session_state.messages.append(
                    {"role": "assistant", "content": partial_assistant_text}
                )
        except Exception as e:
            st.error(f"Error calling GPT-4.1: {e}")



