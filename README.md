# StarCoder2-3B Code Repair with RAG

Retrieval-Augmented Generation (inference-only) on a QLoRA fine-tuned StarCoder2-3B model for automated Python bug fixing.

---

## Project Overview

This project extends a previously fine-tuned StarCoder2-3B code repair model with a retrieval mechanism that injects similar bug-fix examples from the training set into the prompt at inference time. No additional training is performed — the model weights remain frozen.

The core idea: when fixing a buggy code snippet, show the model one highly similar bug-fix pair from its training data as a reference, then ask it to fix the target code.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Query: buggy code from test set                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  SentenceTransformer (all-MiniLM-L6-v2)         │
│  Encode query to 384-dim embedding              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  FAISS Index (cosine similarity)                │
│  20,077 training buggy code embeddings          │
│  Return top-1 similar example                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Prompt Construction                            │
│  If similarity > 0.85:                          │
│    [retrieved buggy → fixed pair]               │
│    [target buggy code]                          │
│  Else:                                          │
│    [target buggy code only]                     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  StarCoder2-3B (base + LoRA checkpoint-2200)    │
│  Generate fixed code                            │
│  Stop at <|endoftext|> (token ID 0)             │
└─────────────────────────────────────────────────┘
```

---

## Model Details

| Component | Value |
|-----------|-------|
| Base Model | bigcode/starcoder2-3b |
| Fine-tuning | QLoRA (4-bit NF4, r=16, alpha=16) |
| Checkpoint | Step 2200 (val_loss = 0.170352) |
| Training Data | 20,077 Python bug-fix pairs |
| Prompt Format | `### Buggy Python Code\n{buggy}\n\n### Corrected Python Code\n{fixed}` |
| Stop Token | `<\|endoftext\|>` (ID 0) |

---

## RAG Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| Index Type | FAISS IndexFlatIP (inner product = cosine after L2 normalization) |
| Knowledge Base | 20,077 train samples |
| k (retrieval) | 1 |
| Similarity Threshold | 0.85 (skip RAG below this) |
| Max Sequence Length | 1024 tokens |
| Max New Tokens | 378 |
| Decoding | Greedy (do_sample=False) |

---

## Dataset

Custom hybrid dataset of Python buggy/fixed code pairs.

| Split | Samples |
|-------|---------|
| Train (RAG index) | 20,077 |
| Test (evaluation) | 2,510 |

---

## Results

### Full Test Set (2,510 samples)

| Metric | Non-RAG v3 | RAG v3 | Change |
|--------|-----------|--------|--------|
| Exact Match | 16.2% | **21.9%** | **+5.7%** |
| Corpus BLEU | 89.7 | 85.9 | -3.8 |
| ROUGE-1 | 89.0 | 88.3 | -0.7 |
| ROUGE-2 | 82.4 | 81.9 | -0.5 |
| ROUGE-L | 88.4 | 87.8 | -0.6 |
| Edit Ratio (mean) | 0.840 | 0.834 | -0.006 |
| Edit Ratio (median) | 0.874 | 0.871 | -0.003 |

### Key Finding

RAG improves **exact match by +5.7 percentage points** (16.2% → 21.9%) at a slight cost to surface similarity metrics. The retrieved example helps the model produce more precise, targeted fixes — the kind that exactly match the ground truth — even though the overall text is slightly less similar.

This trade-off is favorable for code repair: exact match is the most meaningful metric, and a 35% relative improvement is significant.

---

## Progression Across Versions

| Version | Approach | Test EM | BLEU |
|---------|----------|---------|------|
| v1 | QLoRA, broken loss masking | 0.0% | 13.9 |
| v2 | Fixed loss masking, 1k subset | 1.5% | 32.9 |
| v3 | Stop token, full 20k dataset | 16.2% | 89.7 |
| v3 + RAG | Inference-time retrieval | **21.9%** | 85.9 |

---

## Files

| File | Description |
|------|-------------|
| `knowledge_base.json` | Retrieved examples metadata (20,077 entries) |
| `code_index.faiss` | FAISS vector index for fast similarity search |
| `rag_v3_full_results.csv` | Full test predictions with all metrics |
| `checkpoint-2200/` | LoRA adapter weights (best checkpoint) |

---

## Reproduce

1. **Load model** — base `bigcode/starcoder2-3b` + LoRA adapter `checkpoint-2200`
2. **Build RAG index** — embed train `buggy_code` with `all-MiniLM-L6-v2`, store in FAISS
3. **At inference** — retrieve top-1 similar example, prepend to prompt if similarity > 0.85
4. **Generate** — greedy decoding with stop token `<|endoftext|>`

---

## Limitations

- Retrieval quality depends on embedding model; `all-MiniLM-L6-v2` may not capture code semantics optimally
- Similarity threshold (0.85) was chosen heuristically — could be tuned
- Only top-1 retrieval tested; multiple examples hurt performance in early experiments
- The "other" bug type (62% of data) still has low exact match
