# evo2-cpu-offload

Run the **Evo2‑7B** genomic foundation model (StripedHyena) on a single
**12 GB consumer GPU** — an RTX 3060 — by splitting its 32 blocks across GPU
and CPU and streaming weights per block. One file, no framework: importing
`evo2_backend` patches the upstream `vortex` model in place and gives you a
small backend class for embeddings, sequence scoring, variant effect, SAE
features, and generation.

Built for the **ViDaB‑Embed B0.1** project (does Evo2's embedding space encode
RNA‑virus taxonomy on out‑of‑distribution coral RdRp?). Most of it was written
with coding agents and then checked against a clean full‑GPU run on vast.ai. It
is shared as a working artifact, not a polished library.

## Why

Evo2‑7B doesn't fit in 12 GB of VRAM. The usual answers (a bigger card, or a
rented A100/3090) defeat the point of working locally. This makes the model
*usable* on consumer hardware — slow, but correct enough to generate
embeddings and scores that match a full‑GPU reference run.

## How it works

All of this is applied at **import time** by `evo2_backend.py`:

1. **Block placement.** `StripedHyena.__init__` is monkey‑patched so the first
   `GPU_LAYERS = 19` of the 32 blocks live on `cuda:0` and the rest on `cpu`.

2. **Per‑block weight streaming.** Rather than run single‑threaded CPU matmuls,
   a CPU‑resident block is *temporarily* moved to the GPU for its forward pass
   and then moved back. Peak VRAM is therefore ~one block (~400 MB) at a time,
   not the whole tail of the network.

3. **CPU‑safe device handling.** `vortex/model/model.py` is patched (idempotently)
   so the `with torch.cuda.device(...)` contexts fall back to
   `contextlib.nullcontext()` when a block is on CPU, and so the projection norm
   and block‑init paths don't assume CUDA.

4. **FP8 disabled.** The RTX 3060 is `sm_8.6`; FP8 needs `sm_8.9+` (Ada/Hopper).
   The loader swaps in a generated `evo2-7b-1m.no-fp8.yml` config so the 1M‑context
   model loads without FP8 projections.

5. **CPU‑attention skip.** Blocks whose mixer is attention can't run on CPU
   (the rotary kernel is Triton/CUDA‑only), so on CPU they're skipped. This is a
   deliberate approximation — see *Caveats*.

## API

```python
from evo2_backend import Evo2Backend

be = Evo2Backend()
be.ensure_loaded()                      # lazy, idempotent

emb = be.extract_embeddings(seq)        # block-26 hidden states [L, 4096] + mean
sc  = be.score_sequence(seq)            # per-position autoregressive log p, perplexity, entropy
var = be.variant_effect(ref, alt)       # Δ log p between equal-length sequences
fr  = be.extract_sae_features(seq)      # Goodfire L26 SAE features (needs the SAE checkpoint)
gen = be.generate(prompt, n_tokens=100) # autoregressive generation (feature steering is experimental)
```

## Performance

On an RTX 3060 (12 GB, bf16) with a Ryzen 2700X / 16 GB DDR4:

| operation | length | time |
|---|---|---|
| forward to block 26 | L = 720 | ~5 min |
| full forward to logits | L = 720 | a few min |
| SAE encode (GPU) | — | < 1 s |

Slow, but it runs — and the embeddings match a full‑GPU vast.ai RTX 3090 run.

## Requirements

- `torch` (CUDA build), `numpy`, `pyyaml`
- the **Evo2** and **vortex** packages from the
  [Arc Institute Evo2 release](https://github.com/ArcInstitute/evo2)
- *(optional, for SAE features)* the Goodfire `Evo-2-Layer-26-Mixed` SAE checkpoint

## Before you run it

This is research code lifted from a working rig, not a packaged tool. You will
need to edit a few constants near the top of `evo2_backend.py`:

- `sys.path.insert(0, …)` and `SAE_PATH` point at absolute paths on the author's
  machine — change them for yours.
- `evo2_backend.py` **rewrites `vortex/model/model.py` in place** (the patches are
  idempotent and guarded, but they do modify your installed package). Work in a
  throwaway virtualenv if that bothers you.
- `GPU_LAYERS = 19` is tuned for 12 GB. Raise it if you have more VRAM, lower it
  if you OOM.

## Caveats

- The **CPU‑attention skip** means a handful of attention mixers in the CPU tail
  are bypassed. For block‑26 embeddings and the scoring used here this matched a
  full‑GPU reference closely, but it is an approximation, not a bit‑exact run.
- Numbers are tuned for one specific card. Treat them as a starting point.

## License

MIT — see [LICENSE](LICENSE).
