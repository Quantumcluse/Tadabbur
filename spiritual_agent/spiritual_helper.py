import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle # Import the pickle module

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\.env')

# Configure Gemini with secure API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=api_key)

# --- DIRECTLY SPECIFY ABSOLUTE PATHS ---
glove_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\glove.6B.100d.txt"
quran_dataset_path = "C:\\Users\\Natasha\\Tadabbur\\dataset\\Quran_dataset.xlsx"

# Define paths for pickle files
glove_pickle_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\glove_embeddings.pkl"
faiss_pickle_path = "C:\\Users\\Natasha\\Tadabbur\\spiritual_agent\\quran_faiss_index.pkl"
# --- END ABSOLUTE PATHS ---

# Load Quran dataset
df = pd.read_excel(quran_dataset_path)
verses = df["Verse"].tolist()

# Load GloVe embeddings
def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def text_to_glove_vector(text, embeddings, dim=100):
    words = text.lower().split()
    valid_vecs = [embeddings[w] for w in words if w in embeddings]
    if not valid_vecs:
        return np.zeros(dim, dtype='float32')
    avg_vec = np.mean(valid_vecs, axis=0)
    norm = np.linalg.norm(avg_vec)
    return avg_vec / norm if norm > 0 else avg_vec

embedding_dim = 100

# --- Load or create GloVe embeddings ---
if os.path.exists(glove_pickle_path):
    print(f"Loading GloVe embeddings from {glove_pickle_path}...")
    with open(glove_pickle_path, 'rb') as f:
        embeddings = pickle.load(f)
    print("GloVe embeddings loaded from pickle.")
else:
    print(f"Loading GloVe embeddings from {glove_path} and saving to pickle...")
    embeddings = load_glove_embeddings(glove_path)
    with open(glove_pickle_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print("GloVe embeddings saved to pickle.")

# --- Load or create FAISS index and verse vectors ---
if os.path.exists(faiss_pickle_path):
    print(f"Loading FAISS index and verse vectors from {faiss_pickle_path}...")
    with open(faiss_pickle_path, 'rb') as f:
        data = pickle.load(f)
        verse_vectors = data['verse_vectors']
        index = data['faiss_index']
    print("FAISS index and verse vectors loaded from pickle.")
else:
    print("Vectorizing verses and building FAISS index, then saving to pickle...")
    verse_vectors = np.array([text_to_glove_vector(v, embeddings, embedding_dim) for v in verses]).astype('float32')
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(verse_vectors)
    
    with open(faiss_pickle_path, 'wb') as f:
        pickle.dump({'verse_vectors': verse_vectors, 'faiss_index': index}, f)
    print("GloVe embeddings with dataset indexed and saved to pickle.")

sid = SentimentIntensityAnalyzer()

def is_comforting(verse):
    scores = sid.polarity_scores(verse)
    return scores['compound'] >= 0

# Get spiritual prompt from user
query = input("Kindly Type your Spirtual Prompt: ").strip()
query_vec = text_to_glove_vector(query, embeddings, embedding_dim).reshape(1, -1).astype('float32')

# Handle unknown queries with no valid GloVe words
if np.all(query_vec == 0):
    print("\nSorry, we couldn't understand your query. Try using simpler English.")
    exit()

# Perform FAISS search
k = 10
distances, indices = index.search(query_vec, k)
best_similarity = distances[0][0]
print(f"Best matched similarity: {best_similarity:.4f}")

# Filter top comforting verses
filtered_indices = [i for i in indices[0] if is_comforting(verses[i])]
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
model = genai.GenerativeModel('models/gemini-1.5-flash')
response = model.generate_content(prompt)

# Handle potential weak replies
if "I’m here to provide spiritual guidance" in response.text or len(response.text.strip()) < 20:
    print("\n✨ Spiritual Helper Says:\n")
    print("Please rephrase your question with more spiritual context. I’m here to help with Quranic insight in English only.")
else:
    print("\n✨ Spiritual Helper Says:\n")
    print(response.text)
