#Library
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(df):
    df=df.copy()
    #1. Feature Engineering
    ponctuation = [".", "!", "?", "accident", "Accident", "PM"]

    for p in ponctuation:
        df[f"has_{p}"] = df["text"].str.contains(p, regex=False)
        
    df["has_url"] = df["text"].str.contains(r"https?://", regex=True)
    df["url_is_https"] = df["text"].str.contains(r"https://", regex=True)
    df["has_CAP"] = df["text"].str.isupper()
    
    #2. Specific to NLP : Lemmatization, Vectorization, TF-IDF
    #Lemmatization and 'normalisation of words'
    nlp = spacy.load("en_core_web_md")

    df["tokens"] = [
    [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    for doc in nlp.pipe(df["text"], batch_size=50)]
    df["texte_clean"] = df["tokens"].apply(lambda toks: " ".join(toks))
    return df