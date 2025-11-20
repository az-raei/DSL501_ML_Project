from google.colab import drive
drive.mount('/content/drive')

import os
save_dir = "/content/drive/MyDrive/hygieia/data/Qwen2.5_Hinglish_Empathy"
os.makedirs(save_dir, exist_ok=True)
print("Saving to:", save_dir)

!pip install -q transformers accelerate bitsandbytes sentencepiece tqdm

from huggingface_hub import login
login(token="hf_token")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Model loaded:", model_name)

def generate_dialogue(topic="self_doubt", turns=4):

    system_prompt = (
        "You are an empathetic Indian mental-health counselor. "
        "Speak in natural Hinglish. Be warm, concise, emotionally intelligent. "
        "Do NOT generate news articles, ads, lists, or analysis. ONLY dialogue."
    )

    user_prompt = (
        f"Create a {turns}-turn counselling dialogue about '{topic}'.\n"
        "Use the format:\n\n"
        "Client: ...\nTherapist: ...\nClient: ...\nTherapist: ...\n\n"
        "Make it feel culturally realistic and conversational."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",    "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.12,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return text

def clean_dialogue(text):
    if text is None:
        return None

    if "Client:" not in text or "Therapist:" not in text:
        return None

    if len(text.split()) < 25:
        return None

    banned = ["Breaking News", "Headline", "Article", "###", "Summary", "AI model"]
    if any(b in text for b in banned):
        return None

    return text.strip()

import json, random
from tqdm import tqdm

num_dialogues = 2000
batch_size = 200

topics = [
    "self_doubt", "loneliness", "anxiety", "academic_pressure",
    "relationship_issues", "family_conflict", "career_confusion",
    "burnout", "financial_stress", "fear_of_failure"
]

print("Starting generation...\n")

for start in range(0, num_dialogues, batch_size):

    end = min(start + batch_size, num_dialogues)
    batch_data = []

    for i in tqdm(range(start, end)):

        topic = random.choice(topics)
        cleaned = None

        while cleaned is None:
            raw = generate_dialogue(topic)
            cleaned = clean_dialogue(raw)

        batch_data.append({
            "id": i + 1,
            "topic": topic,
            "dialogue": cleaned
        })

    out_path = os.path.join(
        save_dir,
        f"hinglish_dialogues_part_{start//batch_size + 1}.jsonl"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in batch_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print("saved batch:", start//batch_size + 1)

print("\nall dialogues generated successfull")
