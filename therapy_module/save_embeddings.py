import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

def load_quran_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    required_columns = {"Name", "Surah", "Ayat", "Verse"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Excel file must contain columns: Name, Surah, Ayat, Verse")
    return df.dropna(subset=["Verse"]).reset_index(drop=True)

# Load dataset
dataset_path = "C:\\Users\\Natasha\\Tadabbur\\Data\\dataset.xlsx"
df = load_quran_dataset(dataset_path)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Verse'].tolist(), convert_to_tensor=True)

# Save both df and embeddings
joblib.dump((df, embeddings), "quran_embeddings.pkl")
print("✅ Saved dataframe and embeddings to quran_embeddings.pkl")
