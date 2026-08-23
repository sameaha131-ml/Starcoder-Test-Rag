# StarCoder2-3B Code Repair with RAG

Inference-only RAG (Retrieval-Augmented Generation) on a fine-tuned StarCoder2-3B model for Python bug fixing.

## Approach
- Retrieve similar bug-fix pairs from the training set using FAISS + sentence-transformers
- Inject retrieved examples as few-shot context into the prompt
- Generate the fixed code with the fine-tuned StarCoder2 model
- No additional training — model weights frozen

## Setup
- Base model: fine-tuned StarCoder2-3B (merged QLoRA adapter)
- Embedding model: all-MiniLM-L6-v2
- Vector index: FAISS (cosine similarity, 20,077 train samples)
- Retrieval: top-3 similar examples per query
- Max sequence length: 1024 tokens
- Max new tokens: 378
- Stop token: `<|endoftext|>`

## Prompt Format
```
### Buggy Python Code
{buggy_code}

### Corrected Python Code
{fixed_code}
```

For RAG, similar examples are appended as additional buggy/fixed pairs before the target code.

## Dataset
- Train: 20,077 samples (used for RAG index)
- Test: 2,510 samples

## Results
| Metric | Non-RAG (v3) | RAG |
|--------|-------------|-----|
| Exact Match | 16.2% | TBD |
| Corpus BLEU | 89.7 | TBD |
| ROUGE-L | 88.4 | TBD |

## Files
- `knowledge_base.json`: Retrieved examples metadata
- `code_index.faiss`: FAISS vector index
- `rag_results.csv`: Generated fixes with metrics
