# Tadabbur

Tadabbur is a spiritual AI assistant leveraging the Qur'an for emotional support and reflection.

## Files

- `qur'an_therapy.py`: Uses semantic search to recommend verses based on emotional prompts.
- `spiritual_helper.py`: Enhances spiritual queries using Gemini + GloVe embeddings and FAISS.
- `semantic_search.py`: Simple semantic similarity tool using MiniLM and FAISS.

## Dataset Requirement

Make sure you have the dataset file `Qur'an Dataset.xlsx` and optionally GloVe embeddings (`glove.6B.100d.txt`).

## Note
This project requires models like `sentence-transformers` and optionally `google-generativeai`.

