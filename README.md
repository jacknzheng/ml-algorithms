# pretraining

Pretraining architectures implemented from scratch in PyTorch. Each one is built up from
first principles with notes on the reasoning, time/space complexity, and correctness tests
against `F.scaled_dot_product_attention` where a reference exists.

## Layout

```
notebooks/     exploratory implementations, one per topic
pretraining/   package for implementations promoted out of the notebooks
```

## `notebooks/attention.ipynb`

|                                | status                           |
| ------------------------------ | -------------------------------- |
| Dot product attention          | done, verified vs. reference     |
| MHA (looped + einsum)          | done, verified                   |
| MHA with KV cache              | done, prefill == decode verified |
| GQA                            | done, prefill == decode verified |
| MLA                            | done, prefill == decode verified |
| Linear attention (naive)       | done                             |
| Linear attention with chunking | not started                      |

No rotary/positional embeddings yet.

## `notebooks/moe.ipynb`

|                                            | status                                                          |
| ------------------------------------------ | --------------------------------------------------------------- |
| `Expert`, `Router`                         | done                                                            |
| `MoeBlock` — top-k routing, shared experts | done                                                            |
| Router load-balancing loss                 | not started — `full_router_probs` is returned, loss not written |
| `MoeTransformer`                           | in progress — holds an `MLA` + `MoeBlock`, no `forward` yet     |
| `MoeGPT`                                   | in progress — empty `__init__`                                  |
| Latent MoE (Kimi K3)                       | not started                                                     |

MLA is copy-pasted into this notebook rather than imported from `attention.ipynb`, so the
two copies can drift.

## `notebooks/gpt2.ipynb`

Markdown notes only (normalization variants, residual streams, dropout). **No code yet** —
no block, config, or model class.

## `notebooks/algorithms.ipynb`

PEFT / fine-tuning. All stubs: LoRA, DPO, SFT loss. RL section empty.

## `notebooks/puzzles.ipynb`

Scratch space for PyTorch indexing semantics — `nonzero(as_tuple=True)`, `index_add_`.

## `pretraining/`

|                  | status                     |
| ---------------- | -------------------------- |
| `deepseek-v3.py` | not started — comment only |

## Setup

```bash
uv sync
```

Python 3.12. Deps: torch, einops, numpy, nbimporter, nbformat.
