def compute_cds(model, tokenizer, df_hi, df_translated_en):

    #df_hi                = original Hindi
    #df_translated_en     = same texts translated to English

    preds_hi = []
    preds_en = []

    for x, y in zip(df_hi["text"], df_translated_en["text"]):

        hi_inp = tokenizer(x, return_tensors="pt").to(model.device)
        en_inp = tokenizer(y, return_tensors="pt").to(model.device)

        preds_hi.append(int(model(**hi_inp).logits.argmax()))
        preds_en.append(int(model(**en_inp).logits.argmax()))

    drift = sum(p1 != p2 for p1,p2 in zip(preds_hi, preds_en)) / len(preds_hi)
    return drift
