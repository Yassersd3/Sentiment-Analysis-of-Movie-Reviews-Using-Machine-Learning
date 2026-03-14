import pandas as pd
import re


df = pd.read_csv("C:\\Users\\yaser\\OneDrive\\IS463-Project\\IMDB Dataset.csv")


df["review"] = df["review"].str.replace(r'<br\s*/?>', ' ', regex=True)


technical_words = {"cinematography", "screenplay", "pacing", "cgi", "acting", "directing", "soundtrack", "editing", "visuals", "script", "cast", "plot"}
rave_words = {"masterpiece", "brilliant", "stunning", "perfect", "superb", "touching", "incredible", "beautiful", "classic", "flawless"}
rant_words = {"garbage", "trash", "unwatchable", "waste", "worst", "awful", "boring", "mess", "disaster", "stupid", "predictable", "cliche"}
recommendation_words = {"recommend", "must see", "must watch", "don't miss", "avoid", "skip", "pass", "worth it"}




df["technical_word_count"] = df["review"].str.count('|'.join(technical_words), flags=re.IGNORECASE)
df["rave_word_count"] = df["review"].str.count('|'.join(rave_words), flags=re.IGNORECASE)
df["rant_word_count"] = df["review"].str.count('|'.join(rant_words), flags=re.IGNORECASE)
df["recommendation_word_count"] = df["review"].str.count('|'.join(recommendation_words), flags=re.IGNORECASE)

df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))
df["exclamation_count"] = df["review"].str.count(r'!') 
df["question_count"] = df["review"].str.count(r'\?')  
df["quote_count"] = df["review"].str.count(r'".*?"')


df["numerical_rating_mentioned"] = df["review"].str.count(r'\d+\s*(?:/|out of)\s*10').astype(int)

print("\n--- Feature Correlations with Sentiment ---")
df["sentiment_code"] = df["sentiment"].astype('category').cat.codes

check_cols = ["technical_word_count", "rave_word_count", "rant_word_count", 
              "recommendation_word_count", "word_count", "exclamation_count", 
              "question_count", "numerical_rating_mentioned", "sentiment_code"]

print(df[check_cols].corr()["sentiment_code"].sort_values(ascending=False))


df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
df.drop(columns=["sentiment_code"], inplace=True)

df.to_csv("movie_reviews_data.csv", index=False)
print("\nFile saved as 'movie_reviews_data.csv'. Ready for training!")