# 🎶 Music Recommendation System

Welcome to the **Music Recommendation System**! This project uses content-based filtering to recommend similar songs based on your selected input. It's built using Python, Streamlit for the frontend, and cosine similarity for recommendation logic.

## 🧠 How It Works

The recommendation engine uses cosine similarity on song feature vectors to suggest music that is similar to the one selected.

### Process:
1. User selects a song from a dropdown.
2. App computes cosine similarity between the selected song and others.
3. Top 5 most similar songs are displayed with their respective artists.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Cosine Similarity
- **Libraries**: `pandas`, `joblib`, `logging`
