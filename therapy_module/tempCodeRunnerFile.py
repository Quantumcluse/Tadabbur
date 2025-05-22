import pandas as pd

def load_quran_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the Qur'an dataset from Excel, ensures necessary columns exist, and removes missing verses.
    """
    df = pd.read_excel("C:\\Users\\Natasha\\Tadabbur\\Data\\dataset.xlsx")
    required_columns = {"Name", "Surah", "Ayat", "Verse"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Excel file must contain columns: Name, Surah, Ayat, Verse")
    return df.dropna(subset=["Verse"]).reset_index(drop=True)

# 🔄 Replace the path below with your actual file location in Drive
dataset_path = "C:\\Users\\Natasha\\Tadabbur\\Data\\dataset.xlsx"
df = load_quran_dataset(dataset_path)
print(f"✅ Loaded {len(df)} verses.")

from sentence_transformers import SentenceTransformer

def load_model_and_embed(df: pd.DataFrame):
    """
    Loads the sentence transformer model and computes embeddings for all verses.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Verse'].tolist(), convert_to_tensor=True)
    return model, embeddings

# 🔁 Run this only once unless you restart the runtime
model, verse_embeddings = load_model_and_embed(df)
print("✅ Model loaded and embeddings generated.")

def define_emotion_prompts() -> dict:
    """
    Maps emotion labels to descriptive queries.
    """
    return {
        "hopeless": "feeling of hopelessness and needing hope from Allah",
        "anxious": "anxiety and fear, seeking calm and peace from Allah",
        "depressed": "depression and sadness, needing emotional healing from Allah",
        "fear": "fearful heart, looking for Allah’s protection and strength",
        "lonely": "feeling alone, seeking the companionship of Allah",
        "angry": "anger and frustration, needing patience and control",
        "grateful": "deep gratitude and thankfulness to Allah",
        "lost": "feeling lost and directionless, seeking guidance from Allah",
        "guilty": "guilt and remorse, asking for repentance and mercy",
        "hopeful": "hope for better days, and trust in Allah’s plan"
    }

emotion_prompts = define_emotion_prompts()

from sentence_transformers import util
import numpy as np

def retrieve_one_relevant_verse(df: pd.DataFrame, embeddings, model, query: str, top_k: int = 15) -> pd.Series:
    """
    Returns one randomly chosen relevant verse from the top_k semantically similar matches.
    """
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

# 👇 Select your emotion and get one verse
print("\n📖 Qur'an Emotional Assistant")
emotion = input(f"What emotion are you feeling? ({', '.join(emotion_prompts.keys())}): ").strip().lower()

if emotion not in emotion_prompts:
    print("❌ Invalid emotion. Please choose from the list.")
else:
    query = emotion_prompts[emotion]
    result = retrieve_one_relevant_verse(df, verse_embeddings, model, query)

    print("\n🕋 Verse from the Qur'an:\n")
    print(f"📌 Surah {result['Name']} ({int(result['Surah'])}), Ayah {int(result['Ayat'])}:\n{result['Verse']}\nScore: {result['Score']:.3f}")



