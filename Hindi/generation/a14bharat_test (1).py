from google.colab import drive
drive.mount('/content/drive')

import os
save_dir = "/content/drive/MyDrive/hyigiea/data/Hinglish_Empathy_Dataset"
os.makedirs(save_dir, exist_ok=True)
print(f"output directory: {save_dir}")

from huggingface_hub import login
login("huggingface_api")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "ai4bharat/indic-gpt"

tokenizer = AutoTokenizer.from_pretrained("aashay96/indic-gpt")
model = AutoModelForCausalLM.from_pretrained("aashay96/indic-gpt").to("cuda")

def generate_hinglish_dialogue(topic="self_doubt", turns=5):
    prompt = f"""
Generate an empathetic counseling dialogue between a counselor and a client.
The dialogue should be in natural Hindi-English mix (Hinglish),
like how people in India actually talk. Keep tone emotionally intelligent and culturally grounded.

Example style:
Client: Mujhe lagta hai main kisi kaam ka nahi hoon...
Counselor: Aapka yeh mehsoos karna valid hai, lekin chalo milke samajhte hain ki aapko aisa kyun lagta hai.

Now generate a {turns}-turn dialogue on the topic of {topic}.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=300,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()  # freememory after each loop

    return text

import os, json
from tqdm import tqdm

num_dialogues = 2000
batch_size = 200
topics = ["self_doubt", "loneliness", "anxiety", "failure", "relationship_issues"]
save_dir = "/content/drive/MyDrive/hyigiea/data/Hinglish_Empathy_Dataset"
os.makedirs(save_dir, exist_ok=True)

for start in range(0, num_dialogues, batch_size):
    end = min(start + batch_size, num_dialogues)
    batch_data = []

    for i in tqdm(range(start, end)):
        topic = topics[i % len(topics)]
        dialogue = generate_hinglish_dialogue(topic)
        batch_data.append({"id": i+1, "topic": topic, "dialogue": dialogue})

    # save batch incrementally
    output_path = os.path.join(save_dir, f"hinglish_empathy_dialogues_part_{start//batch_size+1}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in batch_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"saved batch {start//batch_size+1}: {len(batch_data)} dialogues")



