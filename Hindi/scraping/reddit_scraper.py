import praw
import json
import time
import re
from tqdm import tqdm

REDDIT_CLIENT_ID = "YOUR_ID"
REDDIT_SECRET = "YOUR_SECRET"
REDDIT_AGENT = "EmpathyDatasetScraper/1.0"

SAVE_PATH = "reddit_dataset.jsonl"

SUBREDDITS = [
    "AmItheKameena",
    "Hindi",
    "India",
    "IndianTeenagers",
    "RelationshipIndia"
]

EMOTION_KEYWORDS = [
    "anxiety", "depression", "sad", "lonely", "stress", "mental",
    "panic", "cry", "breakdown", "family", "parents", "hurt", 
    "relationship", "fight", "lost", "hopeless", "angry", 
    "overwhelmed", "suicide", "scared"
]


reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_AGENT
)

def clean_text(text):
    if text is None:
        return ""

    text = text.replace("\n", " ").strip()

    #links
    text = re.sub(r"http\S+", "", text)

    #remove markdown artifacts
    text = re.sub(r"&gt;|&lt;|&amp;", "", text)

    return text.strip()

def is_emotional(text):
    text_low = text.lower()
    return any(word in text_low for word in EMOTION_KEYWORDS)


def scrape_subreddit(subreddit_name, limit=500):
    subreddit = reddit.subreddit(subreddit_name)
    collected = []

    print(f"\n🔍 Scraping r/{subreddit_name} ...")

    for post in tqdm(subreddit.new(limit=limit)):

        title = clean_text(post.title)
        body = clean_text(post.selftext or "")

        #must contain emotional signals
        if not (is_emotional(title) or is_emotional(body)):
            continue

        #top-level comments
        post.comments.replace_more(limit=0)
        comments = []

        for top_comment in post.comments[:3]:
            text = clean_text(top_comment.body)
            if len(text) < 5:
                continue
            comments.append(text)

        if len(comments) == 0:
            continue

        collected.append({
            "subreddit": subreddit_name,
            "title": title,
            "body": body,
            "comments": comments,
            "upvotes": post.score,
            "id": post.id,
            "url": f"https://reddit.com{post.permalink}"
        })

        time.sleep(0.5)

    return collected

def run_scraper():
    all_data = []

    for sub in SUBREDDITS:
        posts = scrape_subreddit(sub, limit=500)
        all_data.extend(posts)

    print(f"\ntotal collected posts: {len(all_data)}")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for entry in all_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nsaved to {SAVE_PATH}")


if __name__ == "__main__":
    run_scraper()
