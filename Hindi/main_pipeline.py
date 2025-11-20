from src.scraping.reddit_scraper import scrape_subreddit
from src.scraping.reddit_merger import merge_reddit
from src.generation.final_text_generator import generate_synthetic_batch
from src.annotation.annotator import annotate_all
from src.modeling.train_asem_classifier import train_model
from src.evaluation.evaluate_asem import evaluate_model

def main():
    print("STEP 1: scraping reddit")
    scrape_subreddit("India")
    
    print("STEP 2: merging reddit dumps")
    merge_reddit()
    
    print("STEP 3: generating synthetic dialogues")
    generate_synthetic_batch()
    
    print("STEP 4: annotating dataset")
    annotate_all()
    
    print("STEP 5: training empathy classifier")
    train_model()
    
    print("STEP 6: evaluating")
    evaluate_model()

if __name__ == "__main__":
    main()
