#Library
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import sys
spacy.cli.download("en_core_web_md")

def preprocessing(df):
    #1. Feature Engineering
    ponctuation = [".", "!", "?"]

    for p in ponctuation:
        df[f"has_{p}"] = df["text"].str.contains(p, regex=False)
        
    df["has_url"] = df["text"].str.contains(r"https?://", regex=True)
    df["url_is_https"] = df["text"].str.contains(r"https://", regex=True)
    df["has_CAP"] = df["text"].str.isupper()
    
    #Lemmatization and 'normalisation of words'
    nlp = spacy.load("en_core_web_md")

    df["tokens"] = [
    [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    for doc in nlp.pipe(df_train["text"], batch_size=50)]
