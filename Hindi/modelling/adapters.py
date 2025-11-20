# models/adapters.py
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, hidden_size=768, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq, hidden]
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        z = self.dropout(z)
        return x + z  # residual

def add_adapters_to_xlm(model, bottleneck=64):
    # model: XLMRobertaModel
    for i, layer in enumerate(model.encoder.layer):
        # XLMRobertaModel -> encoder -> layer -> [i] -> output / attention / ...
        # safe insertion point: after feed-forward (output dense) or before layernorm
        try:
            ff_out = layer.output
            # insert adapter after ff_out
            adapter = Adapter(hidden_size=layer.output.dense.out_features, bottleneck=bottleneck)
            layer.output.adapter = adapter
        except Exception:
            # fallback: place into layer.intermediate if available
            try:
                layer.adapter = Adapter(hidden_size=model.config.hidden_size, bottleneck=bottleneck)
            except Exception:
                print(f"Failed to insert adapter in layer {i}")
    return model
