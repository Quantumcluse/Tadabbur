import streamlit as st


st.set_page_config(page_title="Quranic AI App", layout="wide")

import streamlit as st

# Define the navigation
app = st.navigation([
    st.Page("Pages/home.py", title="Homepage"),
    st.Page("Pages/Semantic.py", title="Semantic Search"),
    st.Page("Pages/therapy.py", title="Quran Therapy"),
    st.Page("Pages/Assisstant.py", title="AI Assistant")
])

# Run the selected page
app.run()

