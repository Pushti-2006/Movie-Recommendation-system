import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Automatically create the pickle files if they don't exist
if not os.path.exists("cosine_sim.pkl") or not os.path.exists("df_cleaned.pkl"):
    df = pd.read_csv("movies.csv")
    df = df[["genres", "keywords", "overview", "title"]].dropna().reset_index(drop=True)
    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]
    df["cleaned_text"] = df["combined"].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    joblib.dump(df, "df_cleaned.pkl")
    joblib.dump(cosine_sim, "cosine_sim.pkl")
else:
    df = joblib.load("df_cleaned.pkl")
    cosine_sim = joblib.load("cosine_sim.pkl")



# app.py
import json
import streamlit as st
from recommend import df, recommend_movies
from omdb_utils import get_movie_details


config = json.load(open("config.json"))

# OMDB api key
OMDB_API_KEY = config["OMDB_API_KEY"]

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommender")

# Using 'title' instead of 'song' now
movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("🎬 Select a movie:", movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)
        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")
        else:
            st.success("Top similar movies:")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster, width=100)
                        else:
                            st.write("❌ No Poster Found")
                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(f"*{plot}*" if plot != "N/A" else "_Plot not available_")
