# pretraining

ML architectures implemented from scratch in PyTorch, as Jupyter notebooks. Each one is
built up from first principles with notes on the reasoning, time/space complexity, and
correctness tests against `F.scaled_dot_product_attention` where a reference exists.

## `attention.ipynb`

| | status |
|---|---|
| Dot product attention | done, verified vs. reference |
| MHA (looped + einsum) | done, verified |
| MHA with KV cache | done, prefill == decode verified |
| GQA | done, prefill == decode verified |
| MLA | done, prefill == decode verified |
| Linear attention (naive) | done |
| Linear attention with chunking | not started |

No rotary/positional embeddings yet.

## `moe.ipynb`

| | status |
|---|---|
| `Expert`, `Router` | done |
| `MoeBlock` — top-k routing, shared experts | done |
| Router load-balancing loss | not started — `full_router_probs` is returned, loss not written |
| `MoeGPT` | in progress — holds an `MLA` + `MoeBlock`, no `forward` yet |
| Latent MoE (Kimi K3) | not started |

## `gpt2.ipynb`

Markdown notes only (normalization variants, residual streams, dropout). **No code yet** —
no block, config, or model class.

## `algorithms.ipynb`

PEFT / fine-tuning. All stubs: LoRA, DPO, SFT loss. RL section empty.

## `puzzles.ipynb`

Scratch space for PyTorch indexing semantics — `nonzero(as_tuple=True)`, `index_add_`.

## Cross-notebook imports

`moe.ipynb` imports from `attention.ipynb` via `nbimporter`. That library ships a legacy
`find_module()` finder that Python 3.12 removed, so the top of the `MoeGPT` cell installs a
small `find_spec` shim to make it work. `nbimporter` only executes `def`/`class`/`import`
statements, so test code in the imported notebook does not run.

## Setup

```bash
uv sync
```

Python 3.12. Deps: torch, einops, numpy, nbimporter, nbformat.
