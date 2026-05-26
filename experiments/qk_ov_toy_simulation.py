#!/usr/bin/env python3

"""
Minimal toy simulation of one attention head to make QK vs OV concrete.

We intentionally use tiny hand-crafted vectors so we can see:
1) QK chooses which source token to attend to.
2) OV decides what information is written to the destination.

This is a pedagogical toy, not a real pretrained model.
"""

from math import exp

TOKENS = ["The", "capital", "of", "France", "is", "___"]
DEST_INDEX = 5  # prediction position ("___")

# Residual vectors: [country_feature, filler_feature]
RESIDUAL = [
    [0.0, 0.2],  # The
    [0.2, 0.7],  # capital
    [0.0, 0.1],  # of
    [1.0, 0.1],  # France
    [0.1, 0.3],  # is
    [0.8, 0.5],  # ___ (query context: "need a country-related source")
]

# QK projection vectors (single-head, scalar q/k for clarity).
WQ = [1.0, 0.0]  # query mainly cares about country_feature
WK = [1.0, 0.0]  # keys expose country_feature

# OV projection vectors.
# WV extracts a scalar "country-ness" from the source.
WV = [1.0, 0.0]
# WO writes into a 3D "vocab-logit space": [Paris, London, Other]
# Positive value boosts Paris when country-ness source is attended.
WO = [3.0, 0.2, 0.1]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    max_x = max(xs)
    exps = [exp(x - max_x) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def format_vec(v, digits=3):
    return "[" + ", ".join(f"{x:.{digits}f}" for x in v) + "]"


def run_toy_head():
    dest_residual = RESIDUAL[DEST_INDEX]
    q = dot(WQ, dest_residual)  # scalar query

    # QK: score every source token for destination "___"
    keys = [dot(WK, r) for r in RESIDUAL]  # scalar keys
    scores = [q * k for k in keys]
    attn = softmax(scores)

    # OV: read value from each source, then weighted sum, then output projection
    values = [dot(WV, r) for r in RESIDUAL]  # scalar values
    mixed_value = sum(a * v for a, v in zip(attn, values))
    logits_update = [w * mixed_value for w in WO]

    return {
        "q": q,
        "keys": keys,
        "scores": scores,
        "attn": attn,
        "values": values,
        "mixed_value": mixed_value,
        "logits_update": logits_update,
    }


def print_table(title, rows):
    print(f"\n{title}")
    for row in rows:
        print(row)


def main():
    print("QK/OV Toy Simulation: The capital of France is ___")
    print(f"Tokens: {' | '.join(TOKENS)}")
    print(f'Destination token: "{TOKENS[DEST_INDEX]}"')

    out = run_toy_head()

    print_table(
        "QK scores and attention weights (where to look):",
        [
            f"{tok:<8} key={out['keys'][i]:.3f}  score={out['scores'][i]:.3f}  attn={out['attn'][i]:.3f}"
            for i, tok in enumerate(TOKENS)
        ],
    )

    print_table(
        "OV source values (what can be copied):",
        [
            f"{tok:<8} value={out['values'][i]:.3f}  weighted={(out['values'][i] * out['attn'][i]):.3f}"
            for i, tok in enumerate(TOKENS)
        ],
    )

    print(f"\nMixed value at destination: {out['mixed_value']:.3f}")
    print(
        f"Logit update [Paris, London, Other]: {format_vec(out['logits_update'])}"
    )

    print("\nInterpretation:")
    print(
        "- QK gave the highest weight to 'France' because its key matched the destination query."
    )
    print(
        "- OV transformed the attended source signal into a logit update that strongly boosts 'Paris'."
    )
    print(
        "- Same attended source with a different WO would write a different output meaning."
    )


if __name__ == "__main__":
    main()
