!pip install transformers accelerate bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, torch

model_name = "sarvamai/sarvam-2b-v0.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

labeler = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

def annotate_empathy(post, response):
    prompt = f"""
You are an empathy annotator for mental-health conversations.

Classify the RESPONSE relative to the POST into ONE of the 5 categories:

1. Emotional Acknowledgement
2. Validation
3. Reflective Understanding
4. Supportive Reframing
5. Action-oriented Guidance

Return ONLY valid JSON:

{{
  "label": "...",
  "explanation": "..."
}}

POST: {post}
RESPONSE: {response}
    """

    out = labeler(
        prompt,
        max_length=256,
        do_sample=False,
        temperature=0.1
    )[0]["generated_text"]

    # try to extract JSON safely
    start = out.find("{")
    end = out.rfind("}")
    json_str = out[start:end+1]

    try:
        return json.loads(json_str)
    except:
        return {"label": "Unknown", "explanation": "Could not parse JSON"}


labeled = []

with open("/content/drive/MyDrive/hyigiea/data/reddit_merged.jsonl") as f:
    for line in f:
        d = json.loads(line)
        lab = annotate_empathy(d["post"], d["response"])
        d["label"] = lab["label"]
        d["explanation"] = lab["explanation"]
        labeled.append(d)

with open("/content/drive/MyDrive/hyigiea/data/reddit_labeled.jsonl", "w") as f:
    for e in labeled:
        json.dump(e, f, ensure_ascii=False)
        f.write("\n")

print("DONE — labeled", len(labeled))
