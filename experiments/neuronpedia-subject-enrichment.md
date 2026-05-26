# Neuronpedia Subject Enrichment Probe

This experiment uses Neuronpedia feature search to probe the subject-enrichment claim from [MLPs in Transformers](/home/curious/Documents/hermes-projects/learn-mech-interp/src/topics/transformer-foundations/mlps-in-transformers/index.md): before the model extracts a final answer, the subject token's residual state may already contain a broader bundle of subject-linked information.

## Question

For a factual prompt such as `Beats Music is owned by ___`, what information appears to accumulate at the subject token before answer extraction?

## Setup

- Model: `gemma-2-2b`
- Neuronpedia source set: `gemmascope-res-16k`
- Primary tool: `search_features_by_token`
- Anchor prompt: `Beats Music is owned by`

### Token roles

- Subject tokens: `Beats`, `Music`
- Relation tokens: `is`, `owned`, `by`
- Prediction position: immediately after `by`

## Reproducible workflow

1. Run token-level feature search on the anchor prompt.
2. Record top features separately for:
   - subject tokens
   - relation tokens
   - the final relation token nearest the prediction site
3. Label each feature as one of:
   - `subject`
   - `relation`
   - `answer`
   - `noise`
4. Repeat with prompt variants:
   - same subject, different relation
   - different subject, same relation
   - subject-only context
5. Compare what stays stable at the subject token and what changes at relation tokens.
6. Optionally repeat the same prompt at earlier layers to estimate when subject information becomes more semantic.

### MCP calls used

The experiment was run with Neuronpedia's `search_features_by_token` tool using prompts such as:

- `Beats Music is owned by`
- `Beats Music was founded by`
- `Instagram is owned by`
- `Beats Music`

And layer-specific probes such as:

- `5-gemmascope-res-16k`
- `10-gemmascope-res-16k`
- `20-gemmascope-res-16k`

## Results

### 1. Anchor prompt: `Beats Music is owned by`

| Token | Feature | Explanation | Role | Confidence |
| --- | --- | --- | --- | --- |
| `Beats` | `16335` | `beat` | noise / lexical | strong |
| `Beats` | `6918` | `audio devices for ears` | subject-adjacent | suggestive |
| `Music` | `1407` | `music` / `musical` cluster | subject | strong |
| `Music` | `13291` | `music festivals, fundamentals, videos, teacher` | subject/domain | suggestive |
| `owned` | `15435` | `ownership and owners` | relation | strong |
| `by` | `11259` | `company parent group` | relation-to-answer bridge | strong |
| `by` | `16331` | `Apple iOS iPhone` | answer-linked | strong |

Interpretation:

- The subject token `Music` carries stable music-domain features.
- The relation token `owned` carries explicit ownership features.
- Near the prediction site, the model shows both a generic ownership/company-parent feature and a concrete `Apple`-linked feature.

### 2. Same subject, different relation: `Beats Music was founded by`

| Token | Feature | Explanation | Role | Confidence |
| --- | --- | --- | --- | --- |
| `Music` | `1407` | `music` / `musical` cluster | subject | strong |
| `Music` | `13291` | `music festivals, fundamentals, videos, teacher` | subject/domain | suggestive |
| `was` / `founded` | `8511`, `5395` | `past origin or creation`, `establishment origin` | relation | strong |
| `by` | `16331` | `Apple iOS iPhone` | answer-linked spillover | suggestive |

Interpretation:

- The subject-token features stay largely the same when only the relation changes.
- The relation features switch from ownership to founding/establishment.
- `Apple` still appears near `by`, which suggests either cached association strength or feature-search noise rather than a clean relation-specific answer channel.

### 3. Different subject, same relation: `Instagram is owned by`

| Token | Feature | Explanation | Role | Confidence |
| --- | --- | --- | --- | --- |
| `Instagram` | `3886` | `Instagram posts and accounts` | subject | strong |
| `Instagram` | `13470` | `facebook and twitter content` | subject/domain | strong |
| `Instagram` | `14277` | `Facebook platform and users` | subject/domain | suggestive |
| `owned` | `15435` | `ownership and owners` | relation | strong |
| `by` | `11259` | `company parent group` | relation-to-answer bridge | strong |

Interpretation:

- Changing the subject changes the subject-token features substantially.
- The ownership feature on `owned` remains stable.
- This is the clearest evidence in the probe set that subject and relation information are at least partly separable.

### 4. Subject-only context: `Beats Music`

| Token | Feature | Explanation | Role | Confidence |
| --- | --- | --- | --- | --- |
| `Beats` | `16335` | `beat` | noise / lexical | strong |
| `Beats` | `6918` | `audio devices for ears` | subject-adjacent | suggestive |
| `Music` | `1407` | `music` / `musical` cluster | subject | strong |
| `Music` | `13291` | `music festivals, fundamentals, videos, teacher` | subject/domain | suggestive |

Interpretation:

- The same core `Music`-domain features already appear even without an explicit relation.
- This supports the idea that subject-linked information is present before the relation-specific extraction step.

## Across-layer view for the anchor prompt

### Subject token `Music`

| Layer source | Top feature | Explanation | Takeaway |
| --- | --- | --- | --- |
| `5-gemmascope-res-16k` | `9338` | `references to music and musicians` | early subject representation is strongly lexical/domain-level |
| `10-gemmascope-res-16k` | `2498` | `references to music and musicians` | subject-domain signal remains stable mid-stack |
| `20-gemmascope-res-16k` | `1407` | `music` / `musical` cluster | late subject state still carries domain identity, but no equally strong explicit `Apple` feature at the subject token |

### Relation token `owned`

| Layer source | Top feature | Explanation | Takeaway |
| --- | --- | --- | --- |
| `5-gemmascope-res-16k` | `5782` | `ownership and property` | ownership relation is already explicit early |
| `10-gemmascope-res-16k` | `4768` | `ownership and owners` | relation remains stable through middle layers |
| `20-gemmascope-res-16k` | `15435` | `ownership and owners` | relation signal is still strong late, near extraction |

Interpretation:

- Subject information is visible early and remains stable.
- Relation information is also visible early, but is carried on different tokens with different features.
- In this probe, the clearest answer-linked feature (`Apple`) appears near the end of the relation phrase rather than cleanly on the subject token.

## Residual-stream interpretation

This probe supports the following conservative reading of stage A in the article:

- The subject token appears to hold a stable **entity/domain bundle**.
- That bundle is visible even without the relation phrase.
- Changing the relation does not destroy the subject-domain features.
- Changing the subject changes the subject-domain features while preserving ownership-related relation features.

What appears to be present at the subject token:

- entity/domain identity: `music`, `musicians`, `Instagram posts and accounts`
- platform/company-adjacent information: `Facebook platform and users`, `facebook and twitter content`
- lexical residue: `beat`, `words starting with Be`

What we do **not** see cleanly in this first pass:

- a strong, stable, explicit `Apple` feature already sitting on the `Music` subject token

So the safest conclusion is:

> subject enrichment is clearly visible at the level of domain/entity features, while concrete answer extraction appears later and closer to the prediction site in this Neuronpedia probe.

## Confidence labels

- `strong evidence`
  - stable subject-domain features on `Music`
  - stable ownership features on `owned`
  - changed subject features under subject swap
- `suggestive evidence`
  - company/platform features associated with entity tokens
  - answer-linked `Apple` features near `by`
- `likely noise`
  - purely orthographic or token-fragment features like `beat` / `words starting with Be`

## Limitations

- Neuronpedia feature search gives **feature-level evidence**, not a full symbolic readout of the residual stream.
- Many returned explanations are partially lexical or noisy.
- A top feature on one token does not prove causal importance.
- The failed `search_top_features` call means this write-up relies on token-level search rather than whole-prompt ranking.
- Stronger claims about subject enrichment versus extraction would need causal methods such as activation patching or feature ablation.

## Practical next steps

1. Probe more factual prompts with cleaner ownership relations.
2. Add more layers to the across-layer scan for the subject token.
3. Use feature-detail or activation tools to test whether a candidate answer-linked feature activates directly on the subject token in longer contexts.
4. Pair this descriptive workflow with causal interventions in a future experiment.

## Follow-up: does the model gather `Dr. Dre` information?

This follow-up was added to answer a narrower question:

> If the model enriches the subject `Beats Music`, should we expect it to also gather associated facts such as `Dr. Dre`?

### Extra probes

- `Beats Music was founded by Dr. Dre`
- `Dr. Dre founded Beats Music`
- Neuronpedia explanation search for `Dr. Dre`

### What changed when `Dr. Dre` was explicitly present

When `Dr. Dre` was added to the prompt, Neuronpedia surfaced several new features:

| Token | Feature | Explanation | Role | Confidence |
| --- | --- | --- | --- | --- |
| `Dr` / `.` | `6225` | `Dr. followed by name` | explicit name cue | strong |
| `Dr` / `.` | `8479` | `common titles of respect or office` | explicit title cue | strong |
| `Dre` | `957` | `hip hop artists and collaborators` | Dre-adjacent semantic cue | strong |
| `Dre` | `11486` | `names and pronouns` | generic person/name cue | suggestive |
| `by` in `... founded by Dr. Dre` | `945` | `names after 'by'` | relation-to-name bridge | strong |

Interpretation:

- Dre-related information is definitely representable in Neuronpedia's feature space.
- But in these probes it appears **where the token `Dr. Dre` is explicitly written**, not as a strong hidden attribute already sitting on the `Music` subject token.

### What stayed the same on the subject token

Across these Dre-focused prompts, the `Music` token still mainly showed the same subject/domain features as before:

- `1407`: `music` / `musical` cluster
- `13291`: `music festivals, fundamentals, videos, teacher`

What we did **not** see:

- a clean `Dr. Dre` feature on the `Music` token
- a clean hip-hop / Dre-collaborator feature on the `Music` token

### Strongest conclusion from the Dre follow-up

The safest reading is:

> `gemma-2-2b` can clearly represent `Dr. Dre` when the name is present in context, but this probe still does not show strong evidence that the `Music` subject token alone is carrying a clearly recoverable Dre-specific feature during subject enrichment.

That means the first-pass result should be interpreted as:

- the model **does gather some additional information** beyond the literal surface form, especially domain/entity information
- but our current Neuronpedia workflow does **not yet show broad retrieval of all associated facts** such as `Dr. Dre` from the subject token alone

### Important caution

This is still not proof of absence.

Possible reasons Dre-like information may be missing from the top subject-token features:

- it may be distributed across several weaker features
- it may appear at another layer or position
- it may require a richer context to become salient
- the feature search may prioritize more dominant domain-level features over rarer factual associations

### Additional note from explanation search

A global Neuronpedia explanation search for `Dr. Dre` returned explicit `Dr. Dre` features in larger Gemma-3 model indexes. That is useful as a sanity check that Neuronpedia can represent the concept, but it does not directly answer the question for `gemma-2-2b` subject enrichment.
