import json
import re
import emoji
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    if text is None:
        return ""

    # remove emojis
    text = emoji.replace_emoji(text, replace='')

    # remove weird punctuation repetition
    text = re.sub(r"([.!?])\1+", r"\1", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove Reddit formatting
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")

    return text


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["post"] = clean_text(d.get("post", ""))
            d["response"] = clean_text(d.get("response", ""))
            d["label"] = d.get("label", "Unknown")
            data.append(d)
    return data


def prepare_dataset(reddit_path, synthetic_path):

    reddit = load_jsonl(reddit_path)
    synthetic = load_jsonl(synthetic_path)

    print("Loaded Reddit:", len(reddit))
    print("Loaded Synthetic:", len(synthetic))

    df = pd.DataFrame(reddit + synthetic)

    # drop empty rows
    df = df[(df["post"] != "") & (df["response"] != "")]

    # split
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    print("Train:", len(train), "Val:", len(val), "Test:", len(test))
    return train, val, test

