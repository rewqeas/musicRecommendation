import streamlit as st
from recommendations import get_recommendations,df

#set up the Streamlit app
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="centered",
)

st.title("Music Recommender System 🎶")

song_list = sorted(df['song'].dropna().tolist())

selected_song = st.selectbox("🎵Select a song:", song_list)

if st.button("Get Recommendations"):

    with st.spinner("Fetching recommendations..."):
        recommendations = get_recommendations(selected_song)
        if recommendations.empty:
            st.warning(f"No recommendations found for '{selected_song}'.")
        else:
            st.success(f"Recommendations for '{selected_song}':")
            st.table(recommendations)