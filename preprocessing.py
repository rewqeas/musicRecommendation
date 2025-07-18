
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import nltk#natural language toolkit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


logging.info("Logging is set up for preprocessing.")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')



#load and sample dataset
try:
    df = pd.read_csv('songs.csv').sample(10000)
    logging.info("Dataset loaded successfully.")

except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise e



#drop link column
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

#text cleaning
logging.info("Starting text cleaning.")
stop_words= set(stopwords.words('english'))

def preprocessing_text(text):
  #remove special characters and columns
  # text = re.sub("[%s]" % re.escape(string.punctuation),"",text)only for punctuation'
  text = re.sub(r"[^a-zA-Z\s]","",text)

  #lowercase conversion
  text = text.lower()

  #word tokenization
  tokens = word_tokenize(text)
  tokens = [word for word in tokens if word not in stop_words]
  #seperating words with space as while fiting the model in tfidfVectorizer, each word is counted as a token, so if we combine all words as one, there will be no unique value so model will have no training
  return " ".join(tokens)


#apply changes to lyrics
logging.info("Starting text preprocessing.")
df['cleaned_text'] = df['text'].apply(preprocessing_text)
logging.info("Text cleaning completed.")



# conversion of words to numbers
logging.info("Starting TF-IDF vectorization.")
tdidf_vectorizer = TfidfVectorizer()
tdidf_matrix = tdidf_vectorizer.fit_transform(df['cleaned_text'])
logging.info("TF-IDF vectorization completed.")




# compute cosine similarity
logging.info("Calculating cosine similarity.")
cos_similarity = cosine_similarity(tdidf_matrix,tdidf_matrix)




#saving the cleaned data and cosine similarity matrix
joblib.dump(df,filename='df_cleaned.pkl')
joblib.dump(tdidf_matrix, filename='tdidf_matrix.pkl')
joblib.dump(cos_similarity, filename='cosine_sim.pkl')
logging.info("Data and cosine similarity matrix saved successfully.")

logging.info("Data preprocessing completed and saved successfully.")