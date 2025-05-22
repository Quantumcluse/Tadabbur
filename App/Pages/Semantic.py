import streamlit as st
import pickle
import numpy as np

# Load model, FAISS index, and DataFrame
@st.cache_resource
def load_model_and_index():
    with open("C:\\Users\\Natasha\\Tadabbur\\semantic_engine\\semantic_search_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data['model'], data['index'], data['df']

model, index, df = load_model_and_index()

# Semantic search function using FAISS
def search_semantically(query, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), top_k)
    results = []
    for i in I[0]:
        verse_info = df.iloc[i]
        results.append({
            "verse": f"Surah {verse_info['Name']} ({verse_info['Surah']}), Ayah {verse_info['Ayat']}",
            "text": verse_info['Verse']
        })
    return results

# Streamlit UI
st.title("Semantic Search")
st.write("Search for Quranic verses related to your query using semantic understanding.")

query = st.text_input("Enter your query", label_visibility="visible")

if query:
    st.markdown(f"### Results for: *{query}*")
    results = search_semantically(query)

    if results:
        for res in results:
            st.subheader(res["verse"])
            st.write(res["text"])
            st.markdown("---")
    else:
        st.warning("No relevant results found.")
