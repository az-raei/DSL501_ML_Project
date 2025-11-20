import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positives, temperature=0.1):
    """
    anchor: [B, D]
    positives: [B, K, D] K positive examples per anchor
    Also uses in-batch negatives (other anchors).
    """
    B, D = anchor.shape
    K = positives.shape[1] if positives.dim()==3 else 1
    # reshape positives to [B*K, D]
    pos = positives.view(B*K, D)
    # compute similarities: anchor x all (positives + in-batch negatives)
    # gather all candidates: in-batch (anchors) and explicit positives
    candidates = torch.cat([anchor, pos.view(B, K, D).mean(dim=1)], dim=0)  # simple candidate set
    sim = torch.matmul(anchor, candidates.t()) / temperature  # [B, B+?]
    labels = torch.arange(B, device=anchor.device)
    loss = F.cross_entropy(sim, labels)
    return loss

# in training loop:
# compute embeddings:
# anchor = pooled_anchor  # [B, D]
# positive = pooled_translation  # [B,1,D] or [B,K,D]
# c_loss = info_nce_loss(anchor, positive)
# total_loss = clf_loss + lambda_contrast * c_loss
