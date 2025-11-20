!pip install -q indictrans2 sentencepiece protobuf

from indictrans2 import IndicTransliterator, IndicProcessor
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

bt_hi_model_name = "ai4bharat/indictrans2-en-hi"
bt_hi_tok = AutoTokenizer.from_pretrained(bt_hi_model_name)
bt_hi_model = AutoModelForSeq2SeqLM.from_pretrained(
    bt_hi_model_name, torch_dtype=torch.float16, device_map="auto"
)

bt_en_model_name = "ai4bharat/indictrans2-hi-en"
bt_en_tok = AutoTokenizer.from_pretrained(bt_en_model_name)
bt_en_model = AutoModelForSeq2SeqLM.from_pretrained(
    bt_en_model_name, torch_dtype=torch.float16, device_map="auto"
)

def translate(model, tok, text, src_lang, tgt_lang):
    prompt = f"{src_lang}2{tgt_lang}: {text}"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def back_translate_english(text):
    """ Hinglish → English → Hinglish (Qwen2.5) """

    # Hinglish → English
    messages = [
        {"role": "system", "content": "Translate Hinglish into fluent English."},
        {"role": "user", "content": text}
    ]
    inp = tok.apply_chat_template(messages, return_tensors="pt").to(model.device)
    out = model.generate(inp, max_new_tokens=200, temperature=0.7)
    english = tok.decode(out[0], skip_special_tokens=True)

    # English → Hinglish
    messages2 = [
        {"role": "system", "content": "Translate this English text into natural Hinglish (Hindi-English mix)."},
        {"role": "user", "content": english}
    ]
    inp2 = tok.apply_chat_template(messages2, return_tensors="pt").to(model.device)
    out2 = model.generate(inp2, max_new_tokens=200, temperature=0.7)
    hinglish_back = tok.decode(out2[0], skip_special_tokens=True)

    return hinglish_back.strip()

dialogue = None
while dialogue is None:
    raw = generate_dialogue(topic)
    dialogue = clean_dialogue(raw)

bt_en = back_translate_english(dialogue)
bt_hi = back_translate_hinglish(dialogue)

batch_data.append({
    "id": i + 1,
    "topic": topic,
    "dialogue_original": dialogue,
    "dialogue_bt_en": bt_en,
    "dialogue_bt_hi": bt_hi
})
