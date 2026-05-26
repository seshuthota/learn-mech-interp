#!/usr/bin/env node

/**
 * Minimal toy simulation of one attention head to make QK vs OV concrete.
 *
 * We intentionally use tiny hand-crafted vectors so we can see:
 * 1) QK chooses which source token to attend to.
 * 2) OV decides what information is written to the destination.
 *
 * This is a pedagogical toy, not a real pretrained model.
 */

const TOKENS = ["The", "capital", "of", "France", "is", "___"];
const DEST_INDEX = 5; // prediction position ("___")

// Residual vectors: [country_feature, filler_feature]
const residual = [
  [0.0, 0.2], // The
  [0.2, 0.7], // capital
  [0.0, 0.1], // of
  [1.0, 0.1], // France
  [0.1, 0.3], // is
  [0.8, 0.5], // ___ (query context: "need a country-related source")
];

// QK projection vectors (single-head, scalar q/k for clarity).
const WQ = [1.0, 0.0]; // query mainly cares about country_feature
const WK = [1.0, 0.0]; // keys expose country_feature

// OV projection vectors.
// WV extracts a scalar "country-ness" from the source.
const WV = [1.0, 0.0];
// WO writes into a 3D "vocab-logit space": [Paris, London, Other]
// Positive value boosts Paris when country-ness source is attended.
const WO = [3.0, 0.2, 0.1];

function dot(a, b) {
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

function softmax(xs) {
  const maxX = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - maxX));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

function formatVec(v, digits = 3) {
  return `[${v.map((x) => x.toFixed(digits)).join(", ")}]`;
}

function runToyHead() {
  const destResidual = residual[DEST_INDEX];
  const q = dot(WQ, destResidual); // scalar query

  // QK: score every source token for destination "___"
  const keys = residual.map((r) => dot(WK, r)); // scalar keys
  const scores = keys.map((k) => q * k);
  const attn = softmax(scores);

  // OV: read value from each source, then weighted sum, then output projection
  const values = residual.map((r) => dot(WV, r)); // scalar values
  const mixedValue = attn.reduce((sum, a, i) => sum + a * values[i], 0);
  const logitsUpdate = WO.map((w) => w * mixedValue);

  return { q, keys, scores, attn, values, mixedValue, logitsUpdate };
}

function printTable(title, rows) {
  console.log(`\n${title}`);
  rows.forEach((r) => console.log(r));
}

function main() {
  console.log("QK/OV Toy Simulation: The capital of France is ___");
  console.log(`Tokens: ${TOKENS.join(" | ")}`);
  console.log(`Destination token: "${TOKENS[DEST_INDEX]}"`);

  const out = runToyHead();

  printTable(
    "QK scores and attention weights (where to look):",
    TOKENS.map(
      (tok, i) =>
        `${tok.padEnd(8)} key=${out.keys[i].toFixed(3)}  score=${out.scores[i].toFixed(3)}  attn=${out.attn[i].toFixed(3)}`
    )
  );

  printTable(
    "OV source values (what can be copied):",
    TOKENS.map(
      (tok, i) => `${tok.padEnd(8)} value=${out.values[i].toFixed(3)}  weighted=${(out.values[i] * out.attn[i]).toFixed(3)}`
    )
  );

  console.log(`\nMixed value at destination: ${out.mixedValue.toFixed(3)}`);
  console.log(
    `Logit update [Paris, London, Other]: ${formatVec(out.logitsUpdate)}`
  );

  console.log("\nInterpretation:");
  console.log("- QK gave the highest weight to 'France' because its key matched the destination query.");
  console.log("- OV transformed the attended source signal into a logit update that strongly boosts 'Paris'.");
  console.log("- Same attended source with a different WO would write a different output meaning.");
}

main();
