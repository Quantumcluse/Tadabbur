import streamlit as st
from homepage import show_homepage
from semantic_search import show_semantic_search
from quran_therapy import show_quran_therapy
from ai_assistant import show_ai_assistant

st.set_page_config(page_title="Quranic AI App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Semantic Search", "Quran Therapy", "AI Assistant"])

# Conditional routing
if page == "Homepage":
    show_homepage()
elif page == "Semantic Search":
    show_semantic_search()
elif page == "Quran Therapy":
    show_quran_therapy()
elif page == "AI Assistant":
    show_ai_assistant()
