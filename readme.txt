# Tadabbur

Tadabbur is a spiritual AI assistant that leverages the Qur'an for emotional and semantic therapy using NLP and LLMs.

## 📁 Folder Structure

- `therapy_module/qur'an_therapy.py`: Emotion-based verse retrieval using sentence transformers.
- `spiritual_agent/spiritual_helper.py`: LLM-powered assistant with spiritual tone, using GloVe + FAISS.
- `semantic_engine/semantic_search.py`: Classic semantic similarity engine for Qur'anic verse retrieval.

## 📂 Data (Required)

Make sure to manually provide:
- `Qur'an Dataset.xlsx` — Qur'anic verses in tabular form.
- `glove.6B.100d.txt` — GloVe pre-trained embeddings.

Place them inside a `data/` folder locally.

## 🧠 Dependencies

Install with:
```bash
pip install -r requirements.txt
