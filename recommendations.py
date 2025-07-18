import joblib
import logging

#set up logging

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendations.log',encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("Logging is set up.")

try:
    df = joblib.load('df_cleaned.pkl')
    # print(df.head())
    cosine_sim = joblib.load('cosine_sim.pkl')
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise e

def get_recommendations(song_name, cosine_sim=cosine_sim, df=df, top_n = 5):

    logging.info(f"Getting recommendations for song: {song_name}")

    idx = df[df['song'].str.lower() == song_name.lower()].index
    if idx.empty:
        logging.warning(f"Song '{song_name}' not found in the dataset.")
        return []
    idx = idx[0]

    #getting similarity scores for the song
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    #getting the song indices
    song_indices = [i[0] for i in sim_scores]   
    logging.info(f"Top {top_n} recommendations found for '{song_name}'.")

    #creating a DataFrame for the recommended songs
    result_df = df[['song','artist']].iloc[song_indices].reset_index(drop=True)

    result_df.index = result_df.index + 1
    result_df.index.name = 'S.No'

    return result_df
