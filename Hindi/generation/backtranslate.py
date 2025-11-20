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


def back_translate_hinglish(text):
    """
    Hinglish → Hindi → Hinglish
    or
    Hinglish → English → Hinglish
    """

    try:
        # Hinglish → Hindi
        hi = translate(bt_hi_model, bt_hi_tok, text, "en", "hi")

        # Hindi → Hinglish (via Qwen2.5)
        messages = [
            {"role": "system", "content": "Translate Hindi text back into natural Hinglish. Sound like a real Indian."},
            {"role": "user",    "content": hi},
        ]
        inp = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        out = model.generate(
            inp,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        hinglish_rev = tokenizer.decode(out[0], skip_special_tokens=True)

        return hinglish_rev

    except Exception as e:
        print("Back-translation failed:", e)
        return text
dialogue = None
while dialogue is None:
    raw = generate_dialogue(topic)
    dialogue = clean_dialogue(raw)

# NEW: add BT-augmented version
bt_dialogue = back_translate_hinglish(dialogue)

batch_data.append({
    "id": i + 1,
    "topic": topic,
    "dialogue_original": dialogue,
    "dialogue_bt": bt_dialogue
})

