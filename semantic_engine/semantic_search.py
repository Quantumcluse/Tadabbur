import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

def initialize_semantic_search():
    # Load dataset
    file_path = "C:\\Users\\Natasha\\Tadabbur\\Data\\dataset.xlsx"
    df = pd.read_excel(file_path)
    df.dropna(subset=["Verse"], inplace=True)

    # Initialize model and embeddings
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    # Check for cached embeddings
    embeddings_file = "verse_embeddings.npy"
    if os.path.exists(embeddings_file):
        print("[INFO] Loading cached embeddings from file...")
        embeddings = np.load(embeddings_file)
    else:
        print("[INFO] Generating embeddings for all verses...")
        embeddings = model.encode(df["Verse"].tolist(), show_progress_bar=True)
        np.save(embeddings_file, embeddings)
        print("[INFO] Embeddings saved for future use.")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    # Save model and index to pickle file
    pickle_file = "semantic_search_model.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'index': index,
            'df': df,
            'embeddings': embeddings
        }, f)
    print(f"[INFO] Model and index saved to {pickle_file}")

    return model, index, df

def load_from_pickle(pickle_file="semantic_search_model.pkl"):
    if os.path.exists(pickle_file):
        print("[INFO] Loading model and index from pickle file...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['index'], data['df']
    else:
        print("[INFO] Pickle file not found, initializing new model...")
        return initialize_semantic_search()

# Main execution
if __name__ == "__main__":
    # Try to load from pickle first
    model, index, df = load_from_pickle()
    
    print("[READY] Semantic Search Initialized!")

    while True:
        query = input("\nEnter your search query (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding).astype('float32'), 5)

        print("\nTop relevant verses:\n")
        for i in I[0]:
            verse_info = df.iloc[i]
            print(f"[Surah {verse_info['Name']} ({verse_info['Surah']}), Ayah {verse_info['Ayat']}]:\n{verse_info['Verse']}\n")