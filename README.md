# Synthetic Empathy Across Cultures: Multilingual Detection of Empathy in Hindi Mental Health Dialogues

## DSL501 Machine Learning Project

This repository contains the implementation of a comprehensive machine learning project focused on empathy detection in Hindi-language mental health conversations. The project integrates the ASEM (Attention-based Sentiment and Emotion Modelling) framework with XLM-R cross-lingual embeddings to build the first multilingual empathy-detection model specifically adapted for Hindi.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Dataset Construction](#dataset-construction)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Overview

This project addresses a critical gap in mental health AI systems. Viz, the lack of culturally appropriate empathy detection for Indian languages, specifically Hindi. While Large Language Models (LLMs) are increasingly deployed in therapeutic and emotionally sensitive contexts, their capacity to express culturally appropriate empathy across language boundaries remains largely unstudied.

**Key Objectives:**
- Develop a multilingual empathy detection model for Hindi mental health dialogues
- Integrate ASEM framework with XLM-R cross-lingual embeddings
- Create culturally validated synthetic training data
- Evaluate performance on authentic Indian digital discourse
- Establish foundational benchmarks for cross-lingual empathy evaluation

**Research Context:**
India shows suicide rates reaching 12.4 per 100,000 people, with suicide remaining among the top two causes of death for Indian youth for over two decades. This humanitarian crisis demands scalable, culturally sensitive mental health solutions.

---

## Motivation

### The Problem

Existing empathy evaluation benchmarks primarily focus on English-language interactions, resulting in a significant representational gap for Hindi speakers and other Indian language communities. Current mental health AI tools are:
- Predominantly monolingual and culturally Western-centric
- Unable to handle code-mixing patterns (Hinglish)
- Insensitive to cultural communication styles
- Missing indirect distress signals which are common in Indian contexts

### Why Simple Translation Isn't Enough

Mental health conversations in India occur within hybrid linguistic spaces characterised by:
- **Code-mixing**: Hindi-English mixing (Hinglish) is the norm
- **Cultural communication styles**: More direct in conflict situations, different validation strategies
- **Regional metaphors**: Culturally specific concepts of emotional well-being
- **Familial context**: Stronger emphasis on family and social context than Western frameworks

### Our Approach

We address these challenges through three complementary approaches:
1. Adapting state-of-the-art empathy evaluation frameworks for multilingual contexts
2. Generating and validating culturally appropriate synthetic training data
3. Evaluating performance on authentic Indian digital discourse from Reddit

---

### Training Pipeline Architecture

```
   ┌────────────────────┐
   │ 1. Reddit Scraping │───► reddit_scraper.py
   └──────────┬─────────┘
              ▼
   ┌───────────────────────┐
   │ 2. Merge + Clean Data │───► reddit_merger.py
   └──────────┬────────────┘
              ▼
   ┌──────────────────────────────┐
   │ 3. Synthetic Hinglish LLM    │───► final_text_generator.py
   │    Dialogues (IndicGPT)      │
   └──────────┬───────────────────┘
              ▼
   ┌─────────────────────────────┐
   │ 4. Annotation + ASEM labels │───► annotator.py, dataset_loader.py
   └──────────┬──────────────────┘
              ▼
   ┌──────────────────────────────┐
   │ 5. Classifier Training       │───► train_asem_classifier.py
   │    (ASEM-XLM-R + Adapters)   │
   └──────────┬───────────────────┘
              ▼
   ┌─────────────────────────────┐
   │ 6. Evaluation               │───► evaluate_asem.py
   │    - F1, precision, recall  │
   └──────────┬──────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 7. Cultural Drift Score (CDS)      │───► cds.py
   │    Ensure consistency across langs │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 8. Uncertainty-aware Inference     │───► uncertainty.py (MC Dropout)
   │    - abstain when unsure           │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 9. Error Analysis + Explanations   │───► error_analysis.py, explanation_generator.py
   └────────────────────────────────────┘


```

### Modified Components

**1. Multilingual Encoder Integration**
- Replaced monolingual BERT with XLM-R for cross-lingual transfer
- Leveraged shared representations across 100 languages
- Fine-tuned entire encoder rather than freezing weights
- Enabled seamless handling of code-mixed Hinglish text

**2. Cultural Adaptation Mechanisms**
- Validation through native Hindi speakers
- Preservation of code-mixing patterns instead of cleaning
- Cultural tone and communication style awareness
- Recognition of indirect distress signals common in Indian contexts

**3. Training Procedure Enhancements**
- Multi-stage training (English pretraining → Hindi fine-tuning → Reddit adaptation)
- Class balancing for empathy category imbalance
- BLEU-based quality filtering for synthetic dialogues
- Reduced learning rate during fine-tuning to prevent catastrophic forgetting

**4. Evaluation Framework Extensions**
- Cross-lingual consistency metrics
- Cultural appropriateness scoring
- Explanation quality assessment
- Human validation by native speakers
- 
---

## Methodology

### 1. Model Architecture: ASEM with XLM-R

**ASEM (Attention-based Sentiment and Emotion Modeling)**
The ASEM framework represents a significant advancement in empathy detection by explicitly modeling the relationship between sentiment, discrete emotions, and empathic quality through specialized attention mechanisms.

**Three Primary Components:**

1. **Contextualized Encoding** (XLM-R)
   - Input text encoded through XLM-R transformer (270M parameters)
   - Produces contextualized representations for each token
   - Cross-lingual capability across 100 languages
   - Shared subword vocabulary handles code-mixing

2. **Parallel Attention Mechanisms**
   - **Sentiment Attention**: Identifies sentiment-bearing tokens (positive/negative valence)
   - **Emotion Attention**: Focuses on emotion-expressing tokens (joy, sadness, anger, fear, etc.)
   - Attention-weighted representations capture affective dimensions

3. **Empathy Classification Head**
   - Concatenates attended sentiment + emotion + context features
   - Feed-forward layers for final prediction
   - Multi-class output (5 empathy categories)

**Key Insight:** Empathy judgment requires attending specifically to emotional and evaluative dimensions, not just semantic content.

### 2. Training Procedure

**Stage 1: English Pretraining**
- Dataset: EPITOME (~10,000 counseling dialogues)
- Purpose: Learn general empathy patterns
- Optimizer: AdamW (LR: 2e-5)
- Batch size: 16
- Epochs: 5 with early stopping
- Fine-tune entire XLM-R encoder

**Stage 2: Hindi Fine-tuning**
- Dataset: ~2,000 synthetic Hindi dialogues
- Purpose: Adapt to language-specific patterns
- Optimizer: AdamW (LR: 1e-5, reduced to prevent forgetting)
- Epochs: 3-5
- Features: Code-mixing, cultural validation strategies

**Stage 3: Reddit Adaptation (Optional)**
- Dataset: ~900 Reddit posts from Indian subreddits
- Purpose: Adapt to informal, authentic discourse
- Carefully limited to avoid overfitting

### 3. Empathy Annotation Schema

Five primary empathy categories:

1. **Emotional Acknowledgment**
   - Explicitly recognizing and naming emotions
   - Example: "मैं समझ सकता हूं कि आप कितने निराश महसूस कर रहे हैं"

2. **Validation**
   - Communicating emotional response is legitimate
   - Example: "आपकी स्थिति में कोई भी ऐसा महसूस करेगा"

3. **Reflective Understanding**
   - Deep comprehension through paraphrasing
   - Connecting to broader patterns
   - Identifying implicit emotions

4. **Supportive Reframing**
   - Alternative perspectives while maintaining empathy
   - Example: "शायद उनके कठोर शब्दों के पीछे उनकी अपनी चिंताएं हैं"

5. **Action-Oriented Guidance**
   - Concrete suggestions with empathic tone
   - Practical resources and next steps

---

## Dataset Construction

### 1. Synthetic Dialogue Generation

**Generation Process:**
- Model: GPT-4
- Output: ~2,000 Hindi counseling dialogues
- Scenarios: Academic stress, family conflict, relationships, career anxiety, social isolation, identity struggles
- Structure: 6-10 turn conversations

**Cultural Features:**
- Code-mixing patterns (Hinglish)
- Urban Indian context
- Both formal and informal registers
- Culturally appropriate support strategies

**Quality Control:**
- Native speaker validation
- BLEU-based filtering (detect repetitive/low-quality)
- Iterative refinement
- Cultural appropriateness scoring

### 2. Reddit-Based Authentic Dataset

**Subreddit Selection:**

1. **r/AmItheKameena** (Moral dilemmas, conflict resolution)
   - Indian analog to r/AmItheAsshole
   - Family conflicts, workplace tensions
   - Cultural-specific conflict patterns

2. **r/Hindi** (Language and identity)
   - Linguistic politics
   - Cultural identity discussion
   - Code-mixing negotiation

3. **r/India** (Socio-cultural stressors)
   - Political anxiety, economic uncertainty
   - Mental health stigma
   - Contemporary social issues

4. **r/IndianTeenagers** (Youth expression)
   - Academic pressure, parental expectations
   - Informal, meme-inflected language
   - Gen-Z communication styles

5. **r/RelationshipAdvice** (Relationship struggles)
   - Parental involvement in relationships
   - Caste/community pressure
   - Long-distance challenges

**Data Collection:**
- Period: January 2023 - August 2025
- Tool: PRAW (Python Reddit API Wrapper)
- Posts: ~900 post-response pairs
- **Preservation**: Code-mixing, slang, informal language maintained

**Why Preserve "Messy" Language:**
Real mental health conversations are multilingual, informal, and culturally embedded. Any system unable to handle this reality will fail in actual deployment.

### 3. Dataset Statistics

- **Synthetic Hindi dialogues**: ~2,000
- **Reddit posts**: ~900 (mixed Hindi-English)
- **Empathy categories**: 5 with explanations
- **Validation**: Native speaker reviewed
- **Features**: Authentic code-mixing preserved

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended: NVIDIA V100 with 32GB memory)
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/az-raei/DSL501_ML_Project.git
cd DSL501_ML_Project/Hindi
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install IndicNLP library:**
```bash
pip install indic-nlp-library
python -m nltk.downloader stopwords
```

5. **Download XLM-R base model:**
```bash
python scripts/download_xlmr.py
```

6. **Setup Reddit API (for data collection):**
```bash
# Create praw.ini file with your Reddit API credentials
cp config/praw.ini.example config/praw.ini
# Edit with your credentials
```

---

## Usage

### 1. Generate Synthetic Dialogues

```bash
python scripts/generate_synthetic_data.py \
    --language hindi \
    --num_dialogues 2000 \
    --scenarios all \
    --output data/raw/synthetic_dialogues/
```

### 2. Scrape Reddit Data

```bash
python scripts/scrape_reddit.py \
    --subreddits AmItheKameena Hindi India IndianTeenagers RelationshipAdvice \
    --start_date 2023-01-01 \
    --end_date 2025-08-31 \
    --output data/raw/reddit_posts/
```

### 3. Train Model

**Full pipeline (all stages):**
```bash
python scripts/train_model.py \
    --config config/training_config.yaml \
    --output models/checkpoints/
```

**Individual stages:**
```bash
# Stage 1: English pretraining
python scripts/train_model.py --stage 1 --epochs 5

# Stage 2: Hindi fine-tuning
python scripts/train_model.py --stage 2 --epochs 3

# Stage 3: Reddit adaptation
python scripts/train_model.py --stage 3 --epochs 2
```

### 4. Evaluate Model

```bash
# Evaluate on synthetic test set
python scripts/evaluate_model.py \
    --model models/best_model/ \
    --test_data data/processed/synthetic_test.json \
    --output results/reports/

# Evaluate on Reddit data
python scripts/evaluate_model.py \
    --model models/best_model/ \
    --test_data data/processed/reddit_eval.json \
    --authentic_discourse \
    --output results/reports/reddit_eval.txt
```

### 5. Make Predictions

```bash
# Single prediction
python scripts/predict.py \
    --model models/best_model/ \
    --text "यार मुझे बहुत tension हो रही है career ko lekar"

# Batch prediction
python scripts/predict.py \
    --model models/best_model/ \
    --input_file data/new_conversations.json \
    --output predictions.json
```

### 6. Error Analysis

```bash
python scripts/analyze_errors.py \
    --predictions results/predictions.json \
    --gold_labels data/processed/reddit_eval.json \
    --output results/reports/error_analysis.md
```

### 7. Using Jupyter Notebooks

```bash
jupyter notebook notebooks/
```
