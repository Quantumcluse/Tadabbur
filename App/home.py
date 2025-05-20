# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np

# Page title
st.title("My First Streamlit App")

# Sidebar
st.sidebar.header("User Input")
user_name = st.sidebar.text_input("Enter your name:")

# Welcome message
if user_name:
    st.write(f"Hello, **{user_name}**! Welcome to your app.")
else:
    st.write("Welcome! Please enter your name in the sidebar.")

# Sample data
st.subheader("Random Data Chart")
df = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# Display chart
st.line_chart(df)

# Optional checkbox
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)
