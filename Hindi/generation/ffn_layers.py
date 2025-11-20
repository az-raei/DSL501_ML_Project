import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("aashay96/indic-gpt").to("cuda")

print("\nmodel architecture =", model.config.model_type)
print("\nextracting FFN layers (MLP layers inside each Transformer block)...\n")

ffn_layers = []

# GPT-style models store blocks in model.transformer.h
for idx, block in enumerate(model.transformer.h):
    print(f"layer {idx}")

    # FFN consists of two linear layers:
    #   fc_in  (hidden_dim → intermediate_dim)
    #   fc_out (intermediate_dim → hidden_dim)
    mlp = block.mlp
    
    print("• FFN Input Layer   :", mlp.c_fc)
    print("• FFN Output Layer  :", mlp.c_proj)

    ffn_layers.append(mlp)

print("\ntotal FFN layers found:", len(ffn_layers))
