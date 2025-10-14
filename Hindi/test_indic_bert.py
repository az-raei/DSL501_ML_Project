from google.colab import drive
drive.mount('/content/drive')

!pip install torch transformers datasets sentencepiece indic-nlp-library
!pip install bert-score sacrebleu wandb openai

!git clone https://github.com/MIRAH-Official/Empathetic-Chatbot-ASEM.git
!git clone https://github.com/facebookresearch/XLM.git
!git clone https://github.com/az-raei/DSL501_ML_Project.git

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForMaskedLM.from_pretrained("ai4bharat/indic-bert")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

def generate_empathy_turn(prompt):
    outputs = fill_mask(prompt)
    return outputs[0]["sequence"]

text = "क्लाइंट: मुझे बहुत चिंता हो रही है। \nकाउंसलर: [MASK] शांत रहिए, आप अकेले नहीं हैं।"
print(generate_empathy_turn(text))

from datasets import Dataset

client_prompts = [
    "मुझे हाल ही में नींद नहीं आ रही है।",
    "मैं अपने परीक्षा परिणाम को लेकर बहुत तनाव में हूँ।",
    "परिवार के झगड़ों से मन बहुत भारी है।",
    "मुझे लगता है कि कोई मेरी बात नहीं समझता।",
    "मेरे दोस्त अब पहले जैसे नहीं रहे।",
    "काम का दबाव बढ़ गया है और मैं थक गया हूँ।"
]

counselor_templates = [
    "काउंसलर: [MASK] मैं आपकी भावना समझ सकता हूँ।",
    "काउंसलर: [MASK] चलिए इस पर बात करते हैं।",
    "काउंसलर: [MASK] यह सामान्य है, थोड़ा समय खुद को दीजिए।",
    "काउंसलर: [MASK] आपकी बात सुनकर अच्छा लगा कि आप खुलकर साझा कर रहे हैं।"
]

data = []
for client in client_prompts:
    for template in counselor_templates:
        data.append({
            "client": client,
            "prompt": template
        })

dataset = Dataset.from_list(data)
dataset

def batch_fill_mask(batch):
    filled = fill_mask(batch["prompt"], top_k=1)
    results = []
    for i, outputs in enumerate(filled):
        if isinstance(outputs, list):  # pipeline returns list per sample
            results.append(outputs[0]["sequence"])
        else:
            results.append(outputs["sequence"])
    batch["filled_prompt"] = results
    return batch

batched_dataset = dataset.map(batch_fill_mask, batched=True, batch_size=8)

def combine_dialogue(batch):
    batch["dialogue"] = [f"क्लाइंट: {c}\n{r}" for c, r in zip(batch["client"], batch["filled_prompt"])]
    return batch

batched_dataset = batched_dataset.map(combine_dialogue)


