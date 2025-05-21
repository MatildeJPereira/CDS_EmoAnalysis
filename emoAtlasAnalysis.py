# pip install git+https://github.com/MassimoStel/emoatlas
# python -m spacy download en_core_web_lg
# python -m spacy download it_core_news_lg
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from emoatlas import EmoScores
import pandas as pd
import json

emos_eng = EmoScores(language="english")
emos_it = EmoScores(language="italian")

emotions = ["anger", "joy", "trust", "sadness", "disgust", "fear", "anticipation", "surprise"]

# Function to extract zscores for each emotion
def extract_emotions(df, col_name, model, prefix):
    return pd.DataFrame({
        f"{prefix}_{emotion}": df[col_name].apply(lambda x: model.zscores(x).get(emotion, 0))
        for emotion in emotions
    })

with open("./static/json/mistral_response.json") as f:
    json_res = json.load(f)

df = pd.DataFrame(json_res)
df["english"] = df["english_response"].apply(lambda x: emos_eng.zscores(x))
df["italian"] = df["italian_response"].apply(lambda x: emos_it.zscores(x))

# Extract emotion scores
df_eng = extract_emotions(df, "english_response", emos_eng, "eng")
df_it = extract_emotions(df, "italian_response", emos_it, "it")

# Combine into one DataFrame
df_emotions = pd.concat([df_eng, df_it], axis=1)
df_emotions.to_csv("df_emotions_full_ds.csv")
print(df_emotions)