import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model and data once
@st.cache_resource
def load_model_and_data():
    df, verse_embeddings = joblib.load("C:\\Users\\Natasha\\Tadabbur\\therapy_module\\quran_embeddings.pkl")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, verse_embeddings, model

df, verse_embeddings, model = load_model_and_data()

# Define emotion prompts
def define_emotion_prompts() -> dict:
    return {
        "Hopeless": "feeling of hopelessness and needing hope from Allah",
        "Anxious": "anxiety and fear, seeking calm and peace from Allah",
        "Depressed": "depression and sadness, needing emotional healing from Allah",
        "Fear": "fearful heart, looking for Allah’s protection and strength",
        "Lonely": "feeling alone, seeking the companionship of Allah",
        "Angry": "anger and frustration, needing patience and control",
        "Grateful": "deep gratitude and thankfulness to Allah",
        "Lost": "feeling lost and directionless, seeking guidance from Allah",
        "Guilty": "guilt and remorse, asking for repentance and mercy",
        "Hopeful": "hope for better days, and trust in Allah’s plan"
    }

emotion_prompts = define_emotion_prompts()

# Function to get relevant verse
def retrieve_one_relevant_verse(df, embeddings, model, query, top_k=15):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_results = similarities.topk(k=top_k)

    indices = top_results.indices.cpu().numpy()
    scores = top_results.values.cpu().numpy()

    random_index = np.random.choice(len(indices))
    selected_idx = indices[random_index]
    selected_score = scores[random_index]

    selected_verse = df.iloc[selected_idx].copy()
    selected_verse["Score"] = selected_score
    return selected_verse

# -------------------------------
# Streamlit App Interface
# -------------------------------
st.title("🕋 Quran Therapy")
st.write("Select an emotion to receive comforting guidance from the Qur'an.")

# Emotion dropdown
emotion = st.selectbox(
    "Choose an emotion",
    options=[""] + list(emotion_prompts.keys()),
    index=0,
    format_func=lambda x: "Select..." if x == "" else x
)

if emotion:
    query = emotion_prompts[emotion]
    result = retrieve_one_relevant_verse(df, verse_embeddings, model, query)

    st.markdown(f"### 💬 Therapy for *{emotion}*")
    st.subheader(f"📌 Surah {result['Name']} ({int(result['Surah'])}), Ayah {int(result['Ayat'])}")
    st.write(result['Verse'])
    st.caption(f"🔍 Semantic Match Score: {result['Score']:.3f}")
