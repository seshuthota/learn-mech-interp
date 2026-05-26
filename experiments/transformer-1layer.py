#!/usr/bin/env python3
"""
Trainable 1-layer transformer on a tiny factual dataset.

Goal:
- Train on prompts like: "The capital of France is ___"
- Predict the correct capital token at the final position.
- Inspect what attention and MLP are doing after training.

This is intentionally compact and educational, not optimized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


COUNTRY_TO_CAPITAL = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "India": "Delhi",
    "Spain": "Madrid",
    "Germany": "Berlin",
}

# Fixed template prompts the model sees during training and inference.
# The {country} placeholder is filled per example.
# "___" is the prediction token — the model must output the capital after seeing it.
PROMPT_TEMPLATE = ["The", "capital", "of", "{country}", "is", "___"]


@dataclass
class Config:
    d_model: int = 32   # residual stream / embedding dimension
    d_mlp: int = 64     # hidden dimension of the MLP layer (expand ratio 2x)
    n_epochs: int = 1200
    lr: float = 0.08
    seed: int = 0


def build_vocab():
    # Collect all unique tokens: structural words, country names, capital names.
    base = {"The", "capital", "of", "is", "___"}
    countries = set(COUNTRY_TO_CAPITAL.keys())
    capitals = set(COUNTRY_TO_CAPITAL.values())
    vocab = sorted(base | countries | capitals)       # deterministic order
    tok_to_id = {t: i for i, t in enumerate(vocab)}  # string -> int
    id_to_tok = {i: t for t, i in tok_to_id.items()}  # int -> string
    return vocab, tok_to_id, id_to_tok


def make_dataset(tok_to_id):
    xs = []
    ys = []
    for country, capital in COUNTRY_TO_CAPITAL.items():
        prompt = [t.format(country=country) if "{country}" in t else t for t in PROMPT_TEMPLATE]
        xs.append([tok_to_id[t] for t in prompt])
        ys.append(tok_to_id[capital])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def init_params(cfg: Config, vocab_size: int, seq_len: int):
    torch.manual_seed(cfg.seed)

    def randn(shape, scale=0.02):
        return scale * torch.randn(shape)

    params = {
        # Token & position embeddings: lookup tables.
        "tok_emb": randn((vocab_size, cfg.d_model), 0.08),
        "pos_emb": randn((seq_len, cfg.d_model), 0.08),

        # Attention projection matrices: Q, K, V each going d_model -> d_model.
        "WQ": randn((cfg.d_model, cfg.d_model), 0.08),
        "WK": randn((cfg.d_model, cfg.d_model), 0.08),
        "WV": randn((cfg.d_model, cfg.d_model), 0.08),
        "WO": randn((cfg.d_model, cfg.d_model), 0.08),

        # MLP: expand d_model -> d_mlp, then contract back.
        "W1": randn((cfg.d_model, cfg.d_mlp), 0.08),
        "b1": torch.zeros(cfg.d_mlp),
        "W2": randn((cfg.d_mlp, cfg.d_model), 0.08),
        "b2": torch.zeros(cfg.d_model),

        # Unembedding: project residual stream back to vocabulary logits.
        "W_U": randn((cfg.d_model, vocab_size), 0.08),
        "b_U": torch.zeros(vocab_size),
    }
    # Enable gradient computation for all parameters.
    for p in params.values():
        p.requires_grad_(True)
    return params


def forward(params, x_ids):
    """
    Single-layer transformer forward pass.

    Shapes:
      B = batch size (6 prompts), T = sequence length (6 tokens), D = d_model (32), V = vocab size

    Args:
      x_ids: integer token IDs, shape [B, T]

    Returns:
      logits_last: logits for the final position only, shape [B, V]
      cache:       intermediate activations for interpretability inspection
    """
    bsz, seq_len = x_ids.shape

    # --- Embedding ---
    # Token embeddings: look up each token ID in a learned table -> [B, T, D]
    # Positional embeddings: add a learned vector per position -> [B, T, D]
    # The result x is the residual stream at the input of the block.
    x = params["tok_emb"][x_ids] + params["pos_emb"][None, :, :]

    # --- Attention sub-layer ---
    # Compute Q, K, V from the same input x (self-attention).
    # x:         [B, T, D]
    # WQ/WK/WV:  [D, D]
    # Each matmul: [B, T, D] @ [D, D] -> [B, T, D]
    q = x @ params["WQ"]
    k = x @ params["WK"]
    v = x @ params["WV"]

    # Scaled dot-product attention scores: Q @ K^T / sqrt(d_model)
    # q: [B, T, D], k: [B, T, D]
    # We transpose the last two dims of k: [B, D, T], then matmul -> [B, T, T]
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(x.shape[-1])

    # Apply causal mask: future positions are masked with -inf -> softmax gives 0.
    # Build mask: [T, T], True means "allowed to attend".
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()   # lower triangular = causal
    attn_scores = attn_scores.masked_fill(~mask[None, :, :], float("-inf"))
    attn_weights = F.softmax(attn_scores, dim=-1)            # [B, T, T] — rows sum to 1

    # Weighted sum of values: each position reads from the positions it attends to.
    # attn_weights: [B, T, T], v: [B, T, D] -> [B, T, D]
    attn_mix = attn_weights @ v
    attn_out = attn_mix @ params["WO"]                       # [B, T, D]

    # Residual connection: add attention output to the stream.
    x1 = x + attn_out

    # --- MLP sub-layer ---
    # First projection: expand from d_model to d_mlp (2x wider), then GELU activation.
    h_pre = x1 @ params["W1"] + params["b1"]                # [B, T, d_mlp]
    h = F.gelu(h_pre)                                        # non-linear

    # Second projection: contract back to d_model.
    mlp_out = h @ params["W2"] + params["b2"]                # [B, T, D]

    # Residual connection: add MLP output to the stream.
    x2 = x1 + mlp_out

    # --- Unembedding ---
    # Project the final residual stream to vocabulary logits.
    logits = x2 @ params["W_U"] + params["b_U"]             # [B, T, V]

    # We only care about the prediction at the last position ("___").
    logits_last = logits[:, -1, :]                           # [B, V]

    # Cache activations for post-training interpretability inspection.
    cache = {
        "attn_weights": attn_weights.detach(),   # attention patterns [B, T, T]
        "h_pre": h_pre.detach(),                 # pre-activation MLP hidden states [B, T, d_mlp]
        "h": h.detach(),                         # post-activation MLP hidden states [B, T, d_mlp]
        "x": x.detach(),                         # input to the block (embeddings) [B, T, D]
        "x1": x1.detach(),                       # post-attention residual stream [B, T, D]
        "x2": x2.detach(),                       # post-MLP residual stream [B, T, D]
    }
    return logits_last, cache


def loss_fn(params, x_ids, y_ids):
    # Cross-entropy loss on the final position only.
    logits_last, _ = forward(params, x_ids)                # [B, V]
    return F.cross_entropy(logits_last, y_ids)


def train_step(params, x_ids, y_ids, lr):
    # Zero gradients from the previous step.
    for p in params.values():
        p.grad = None

    # Forward + backward pass.
    loss = loss_fn(params, x_ids, y_ids)
    loss.backward()

    # SGD update: param -= lr * grad  (no optimizer, manual step).
    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad

    return loss.detach()


def topk_indices(values, k=5):
    k = min(k, values.shape[0])
    idx = values.argsort(descending=True)[:k]
    return [int(i) for i in idx]


def run():
    cfg = Config()
    vocab, tok_to_id, id_to_tok = build_vocab()
    x_ids, y_ids = make_dataset(tok_to_id)
    seq_len = x_ids.shape[1]
    params = init_params(cfg, len(vocab), seq_len)

    # --- Training loop ---
    # Full-batch SGD: we only have 6 examples, so no batching needed.
    for epoch in range(1, cfg.n_epochs + 1):
        loss = train_step(params, x_ids, y_ids, cfg.lr)
        if epoch % 200 == 0 or epoch == 1:
            print(f"epoch={epoch:4d} loss={float(loss):.4f}")

    # --- Evaluation ---
    logits_last, cache = forward(params, x_ids)
    probs = F.softmax(logits_last, dim=-1)
    preds = probs.argmax(dim=-1)
    acc = (preds == y_ids).float().mean()

    print("\nTraining set predictions:")
    for i in range(x_ids.shape[0]):
        prompt_tokens = [id_to_tok[int(t)] for t in x_ids[i]]
        gold = id_to_tok[int(y_ids[i])]
        pred = id_to_tok[int(preds[i])]
        p = float(probs[i, preds[i]])
        print(f"{' '.join(prompt_tokens)} -> pred={pred:<7} gold={gold:<7} p={p:.3f}")
    print(f"train_accuracy={float(acc):.3f}")

    # --- Interpretability: inspect attention patterns ---
    # Which source positions does the "___" token attend to most?
    # Expectation: strong attention to the country token (position 3).
    print("\nAttention from final position (token '___'):")
    final_attn = cache["attn_weights"][:, -1, :]  # [B, T] — last row of each attention matrix
    for i in range(x_ids.shape[0]):
        prompt_tokens = [id_to_tok[int(t)] for t in x_ids[i]]
        attn_vals = [float(v) for v in final_attn[i]]
        row = " | ".join(f"{tok}:{val:.3f}" for tok, val in zip(prompt_tokens, attn_vals))
        print(row)

    # --- Interpretability: inspect MLP hidden units ---
    # Which MLP neurons fire most strongly for each country?
    # Each neuron is a detector; top-5 reveals which are most active.
    print("\nTop MLP activations at final position:")
    h_last = cache["h"][:, -1, :]  # [B, d_mlp] — MLP hidden at the "___" position
    for i in range(x_ids.shape[0]):
        country = id_to_tok[int(x_ids[i, 3])]
        top_idx = topk_indices(h_last[i], k=5)
        top_vals = [float(h_last[i, j]) for j in top_idx]
        pairs = ", ".join(f"{j}:{v:.3f}" for j, v in zip(top_idx, top_vals))
        print(f"{country:<8} -> {pairs}")


if __name__ == "__main__":
    run()
