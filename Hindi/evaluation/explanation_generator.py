def generate_explanation(model, tokenizer, post, response):
    prompt = (
        "Explain why the following response fits an empathy category.\n"
        f"POST: {post}\n"
        f"RESPONSE: {response}\n"
        "Write a 2-3 sentence explanation."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(out[0], skip_special_tokens=True)
