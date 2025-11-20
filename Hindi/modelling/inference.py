def predict_empathy(model, tokenizer, post, response):

    text = post + " [SEP] " + response
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    logits = model(**inp).logits
    label_id = int(logits.argmax())

    return {
        "label": id_to_label[label_id],
        "logits": logits.detach().cpu().tolist()
    }
