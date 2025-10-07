import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ================== SETUP ==================
load_dotenv()  # Load .env file from current directory
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="EchoLung AI", page_icon="🩺")

# ------------------ API Key Handling ------------------
if not api_key:
    st.error("❌ Google API key not found! Please add it to your `.env` file in this folder.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    # Test the API key by listing models
    models = [m.name for m in genai.list_models()]
    if not any("gemini" in m for m in models):
        raise ValueError("Gemini models not accessible with this API key.")
    st.success("✅ Gemini API key loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Invalid or unauthorized API key. Please check your Google AI Studio key.\n\n**Error:** {e}")
    st.stop()

# ------------------ Model Init ------------------
MODEL_NAME = "gemini-2.5-flash-lite"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"⚠️ Failed to load model '{MODEL_NAME}': {e}")
    st.stop()

# ================== INTERFACE ==================
st.title("🩺 EchoLung – AI Chatbot")
st.caption("An AI assistant powered by Gemini for your 2D CNN lung sound classification project")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Type your question here...")

# Chat logic
if user_input:
    st.chat_message("user").write(user_input)
    with st.spinner("Thinking..."):
        try:
            response = model.generate_content(user_input)
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"⚠️ Error generating response: {e}"

    st.chat_message("assistant").write(bot_reply)
    st.session_state.chat_history.append((user_input, bot_reply))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)
