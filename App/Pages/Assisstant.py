import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle # Import the pickle module
from dotenv import load_dotenv # Import load_dotenv

# --- Streamlit App Title and Description ---
st.title("AI Assistant")
st.write("Talk to a chatbot trained on Quranic knowledge.")

# --- Configuration and Setup ---

# Load environment variables from .env file
# Ensure this path is correct for your environment
load_dotenv('C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\.env')

# Configure Gemini with secure API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the app if API key is missing
genai.configure(api_key=api_key)

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
    st.info("Downloaded NLTK 'vader_lexicon'.")

# --- DIRECTLY SPECIFY ABSOLUTE PATHS ---
glove_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\glove.6B.100d.txt"
quran_dataset_path = "C:\\Users\\Natasha\\Tadabbur\\dataset\\Quran_dataset.xlsx"

# Define paths for pickle files
glove_pickle_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\glove_embeddings.pkl"
faiss_pickle_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\quran_faiss_index.pkl"
# --- END ABSOLUTE PATHS ---

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# --- Helper Functions (Cached for performance) ---

@st.cache_resource
def load_quran_dataset_cached(path):
   
    try:
        df = pd.read_excel(path)
       # st.success("Quran dataset loaded.")
        return df
    except FileNotFoundError:
        st.error(f"Error: Quran dataset not found at: {path}. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Quran dataset: {e}")
        st.stop()

@st.cache_resource
def load_glove_embeddings_cached(glove_file_path, pickle_file_path):
    """
    Loads GloVe embeddings. Attempts to load from a pickle file first.
    If pickle file not found, loads from the raw GloVe text file and then saves to pickle.
    Caches the result.
    """
    embeddings = {}
    if os.path.exists(pickle_file_path):
        
        try:
            with open(pickle_file_path, 'rb') as f:
                embeddings = pickle.load(f)
            # st.success("GloVe embeddings loaded from pickle.")
        except Exception as e:
            st.warning(f"Could not load GloVe embeddings from pickle ({e}). Attempting to load from raw file.")
            # Fallback to raw file if pickle fails
            embeddings = _load_glove_from_raw(glove_file_path, pickle_file_path)
    else:
        st.info(f"Loading GloVe embeddings from {glove_file_path} (this may take a moment)...")
        embeddings = _load_glove_from_raw(glove_file_path, pickle_file_path)
    return embeddings

def _load_glove_from_raw(glove_file_path, pickle_file_path):
    """Helper to load GloVe from raw text file and save to pickle."""
    embeddings = {}
    try:
        with open(glove_file_path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype='float32')
                embeddings[word] = vector
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(embeddings, f)
       
    except FileNotFoundError:
        st.error(f"Error: GloVe file not found at: {glove_file_path}. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading GloVe embeddings from raw file: {e}")
        st.stop()
    return embeddings

@st.cache_resource
def setup_faiss_index_cached(df_verses, embeddings_data, embedding_dim_val, pickle_file_path):
    """
    Sets up the FAISS index. Attempts to load from a pickle file first.
    If pickle file not found, vectorizes verses, builds the index, and then saves to pickle.
    Caches the result.
    """
    verses_list = df_verses["Verse"].tolist()
    if os.path.exists(pickle_file_path):
       
        try:
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
                verse_vectors_data = data['verse_vectors']
                faiss_index_data = data['faiss_index']
           # st.success("FAISS index and verse vectors loaded from pickle.")
        except Exception as e:
            st.warning(f"Could not load FAISS index from pickle ({e}). Attempting to build from scratch.")
            verse_vectors_data, faiss_index_data = _build_faiss_from_scratch(verses_list, embeddings_data, embedding_dim_val, pickle_file_path)
    else:
        st.info("Vectorizing verses and building FAISS index (this may take a moment)...")
        verse_vectors_data, faiss_index_data = _build_faiss_from_scratch(verses_list, embeddings_data, embedding_dim_val, pickle_file_path)
    return verse_vectors_data, faiss_index_data

def _build_faiss_from_scratch(verses_list, embeddings_data, embedding_dim_val, pickle_file_path):
    """Helper to build FAISS index from scratch and save to pickle."""
    verse_vectors_data = np.array([text_to_glove_vector(v, embeddings_data, embedding_dim_val) for v in verses_list]).astype('float32')
    faiss_index_data = faiss.IndexFlatIP(embedding_dim_val)
    faiss_index_data.add(verse_vectors_data)

    with open(pickle_file_path, 'wb') as f:
        pickle.dump({'verse_vectors': verse_vectors_data, 'faiss_index': faiss_index_data}, f)
   # st.success("GloVe embeddings with dataset indexed and saved to pickle.")
    return verse_vectors_data, faiss_index_data

def text_to_glove_vector(text, embeddings, dim=100):
    """Converts text to a GloVe vector."""
    words = text.lower().split()
    valid_vecs = [embeddings[w] for w in words if w in embeddings]
    if not valid_vecs:
        return np.zeros(dim, dtype='float32')
    avg_vec = np.mean(valid_vecs, axis=0)
    norm = np.linalg.norm(avg_vec)
    return avg_vec / norm if norm > 0 else avg_vec

def is_comforting(verse):
    """Checks if a verse is comforting based on sentiment analysis."""
    scores = sid.polarity_scores(verse)
    return scores['compound'] >= 0

# --- Main Streamlit App Logic ---

# Load resources using cached functions
with st.spinner("Initializing AI Assistant (this may take a moment on first run)..."):
    df = load_quran_dataset_cached(quran_dataset_path)
    embeddings = load_glove_embeddings_cached(glove_path, glove_pickle_path)
    embedding_dim = 100 # GloVe 6B.100d has 100 dimensions
    verse_vectors, index = setup_faiss_index_cached(df, embeddings, embedding_dim, faiss_pickle_path)

# Initialize session state for AI response if not already present
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""

# User input
query = st.text_input("Kindly Type your Spiritual Prompt:", key="user_query")

if st.button("Get Guidance"):
    if not query:
        st.warning("Please enter a spiritual prompt to get guidance.")
    else:
        with st.spinner("Thinking and searching for guidance..."):
            query_vec = text_to_glove_vector(query, embeddings, embedding_dim).reshape(1, -1).astype('float32')

            # Handle unknown queries with no valid GloVe words
            if np.all(query_vec == 0):
                st.session_state.ai_response = "Sorry, we couldn't understand your query. Try using simpler English."
            else:
                # Perform FAISS search
                k = 10
                distances, indices = index.search(query_vec, k)
                # best_similarity = distances[0][0]
                # st.write(f"Best matched similarity: {best_similarity:.4f}") # Uncomment for debugging similarity

                # Filter top comforting verses
                # Ensure 'verses' is accessible, which it is via df.iloc[i]['Verse']
                filtered_indices = [i for i in indices[0] if is_comforting(df.iloc[i]['Verse'])]
                selected_index = random.choice(filtered_indices) if filtered_indices else indices[0][0]

                selected_verse = df.iloc[selected_index]["Verse"]
                selected_surah = df.iloc[selected_index]["Name"]
                selected_ayah = df.iloc[selected_index]["Ayat"]

                # Prepare Gemini prompt
                prompt = (
                    "You are a wise, spiritually uplifting Islamic advisor responding to a distressed believer.\n\n"
                    f"The user said: '{query}'\n\n"
                    f"Here are 5 Quranic verses that may help:\n" +
                    "\n".join(
                        f"- {df.iloc[i]['Verse']} (Surah {df.iloc[i]['Name']}, Ayah {df.iloc[i]['Ayat']})"
                        for i in indices[0]
                    ) + "\n\n"
                    "Your response should:\n"
                    "- Start with gentle encouragement (e.g. 'Don't lose hope, my friend...')\n"
                    "- Show empathy and align with Islamic spirituality.\n"
                    "- End with this Quranic verse:\n"
                    f"  In the Quran, Allah says in Surah {selected_surah}, Ayah {selected_ayah}: \"{selected_verse}\"\n\n"
                    "Keep your answer in English only, limited to 3–5 lines total. No filler, no generic statements."
                )

                # Generate response
                try:
                    model = genai.GenerativeModel('models/gemini-1.5-flash')
                    response = model.generate_content(prompt)

                    # Handle potential weak replies
                    if "I’m here to provide spiritual guidance" in response.text or len(response.text.strip()) < 20:
                        st.session_state.ai_response = "Please rephrase your question with more spiritual context. I’m here to help with Quranic insight in English only."
                    else:
                        st.session_state.ai_response = response.text
                except Exception as e:
                    st.error(f"Error generating AI response: {e}")
                    st.session_state.ai_response = "An error occurred while getting AI guidance. Please try again."

# Display the AI's response
if st.session_state.ai_response:
    st.markdown("---")
    st.subheader("✨ Spiritual Helper Says:")
    st.info(st.session_state.ai_response)
