# StarCoder2-3B Code Repair with RAG

Retrieval-Augmented Generation (inference-only) on a QLoRA fine-tuned StarCoder2-3B model for automated Python bug fixing.

---

## Project Overview

This project extends a previously fine-tuned StarCoder2-3B code repair model with a retrieval mechanism that injects the single most similar bug-fix example from the training set into the prompt at inference time. No additional training is performed — model weights remain frozen.

The core idea: when fixing a buggy code snippet, retrieve one highly similar bug-fix pair from training data. If similarity exceeds 0.85, prepend it as a reference example. Then ask the model to fix the target code.

---

## Architecture

```
Query: buggy code from test/val set
    ↓
SentenceTransformer (all-MiniLM-L6-v2)
Encode to 384-dim embedding
    ↓
FAISS Index (cosine similarity)
20,077 training buggy code embeddings
Return top-1 similar example
    ↓
If similarity > 0.85:
    [retrieved buggy → fixed pair]
    [target buggy code]
Else:
    [target buggy code only]
    ↓
StarCoder2-3B (base + LoRA checkpoint-2200)
Generate fixed code
Stop at <|endoftext|> (token ID 0)
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
| Index Type | FAISS IndexFlatIP |
| Knowledge Base | 20,077 train samples |
| k (retrieval) | 1 |
| Similarity Threshold | 0.85 |
| Max Sequence Length | 1024 tokens |
| Max New Tokens | 378 |
| Decoding | Greedy |

---

## Dataset

| Split | Samples |
|-------|---------|
| Train (RAG index) | 20,077 |
| Validation | 2,510 |
| Test | 2,510 |

---

## Results

### Validation Set (2,510 samples)

| Metric | Non-RAG v3 | RAG v3 | Change |
|--------|-----------|--------|--------|
| Exact Match | 16.5% | **22.7%** | **+6.2%** |
| Corpus BLEU | 88.8 | 86.4 | -2.4 |
| ROUGE-1 | 89.0 | 88.6 | -0.4 |
| ROUGE-2 | 82.5 | 82.3 | -0.2 |
| ROUGE-L | 88.5 | 88.1 | -0.4 |
| Edit Ratio (mean) | 0.840 | 0.837 | -0.003 |
| Edit Ratio (median) | 0.873 | 0.871 | -0.002 |

### Test Set (2,510 samples)

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

RAG consistently improves exact match by **+5.7% to +6.2%** across both splits, at a small cost to surface similarity metrics (BLEU drops 2-4 points). The retrieved example helps the model produce more precise, targeted fixes that exactly match the ground truth.

For code repair, exact match is the most meaningful metric. A relative improvement of ~35% is significant.

---

## Progression

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
| `knowledge_base.json` | RAG knowledge base (20,077 entries) |
| `code_index.faiss` | FAISS vector index |
| `rag_v3_test_results.csv` | Full test predictions with metrics |
| `rag_v3_val_results.csv` | Full validation predictions with metrics |
| `checkpoint-2200/` | LoRA adapter weights |

---

## Limitations

- Retrieval quality depends on the embedding model; `all-MiniLM-L6-v2` may not capture code semantics optimally
- Similarity threshold (0.85) chosen heuristically — could be tuned
- Only top-1 retrieval tested; multiple examples hurt performance
- BLEU drops slightly — RAG improves precision at cost of surface similarity
- The "other" bug type (62% of data) still has low exact match
