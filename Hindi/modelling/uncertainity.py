import torch
import torch.nn.functional as F
import numpy as np

def enable_dropout(model):
    """Enable dropout during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def mc_predict(model, inputs, t=8):
    model.eval()
    enable_dropout(model)  # keep dropout active
    probs = []
    with torch.no_grad():
        for _ in range(t):
            out = model(**inputs).logits
            p = F.softmax(out, dim=-1).cpu().numpy()
            probs.append(p)
    probs = np.stack(probs, axis=0)  # [t, B, C]
    mean = probs.mean(axis=0)  # [B, C]
    entropy = -np.sum(mean * np.log(mean + 1e-12), axis=-1)  # [B]
    return mean, entropy

# usage:
#inputs = tokenizer(text, return_tensors="pt").to(model.device)
#mean, entropy = mc_predict(model, inputs, t=12)
#if entropy[0] > 0.9:   # threshold tune on val set
#    action = "ABSTAIN -> human review"
#else:
#    pred = mean.argmax(axis=-1)[0]
