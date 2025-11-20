# models/asem_xlmr.py
import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, embeddings, mask):
        # embeddings: [batch, seq, hidden]
        scores = self.attention(embeddings).squeeze(-1)   # [batch, seq]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)           # [batch, seq]
        weighted = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return weighted, weights


class ASEM_XLMR(nn.Module):
    def __init__(self, num_labels=5, hidden_size=768):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.sentiment_attn = AttentionLayer(hidden_size)
        self.emotion_attn = AttentionLayer(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = outputs.last_hidden_state

        sent_vec, _ = self.sentiment_attn(embeddings, attention_mask)
        emo_vec, _ = self.emotion_attn(embeddings, attention_mask)
        pooled = embeddings[:, 0]  # CLS token

        concat = torch.cat([pooled, sent_vec, emo_vec], dim=-1)
        logits = self.classifier(concat)
        return logits
