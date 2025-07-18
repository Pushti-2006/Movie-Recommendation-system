import streamlit as st
from recommend import recommend_movies, df
from omdb_utils import get_movie_details

# Securely load OMDB API key
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")
st.markdown("Upload a movie name below to get top similar movies.")

movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    with st.spinner("Getting recommendations..."):
        recommendations = recommend_movies(selected_movie)
        if recommendations is None or recommendations.empty:
            st.error("No similar movies found.")
        else:
            for _, row in recommendations.iterrows():
                title = row['title']
                plot, poster = get_movie_details(title, OMDB_API_KEY)
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(poster if poster != "N/A" else "https://via.placeholder.com/100", width=100)
                    with col2:
                        st.subheader(title)
                        st.markdown(f"*{plot}*")

