"""
Evo2-7B + Goodfire L26 SAE — reusable backend
==============================================

A single class that loads the StripedHyena-based 7B base model with a
24/32 GPU/CPU split, loads the Goodfire BatchTopK SAE for layer 26, and
exposes a small API for a Gradio UI:

    Evo2Backend()
        .ensure_loaded()               # lazy, idempotent
        .forward(seq) -> Tensor        # [1, L, 4096] (block-26 residual, on CPU)
        .get_logits(seq) -> Tensor     # [1, L, 512] (full forward, on CPU)
        .get_sae_features(h) -> Tensor # [1, L, 32768] (on GPU)
        .score_sequence(seq) -> dict   # autoregressive per-position log p
        .variant_effect(ref, alt) -> dict

All vortex patches (init device split, CPU-safe `with torch.cuda.device`,
F.linear device handling, CPU-attention skip in the truncated loop) are
applied at import time, so importing this module is enough to make the
Evo2 Python package CPU/GPU-friendly.

Performance (RTX 3060, 12 GB VRAM, bf16):
  - forward to block 26, L=720: ~5 min  (CPU on blocks 25–26; block 24
    skipped because its attention mixer needs Triton on CUDA)
  - one full forward to logits, L=720: also a few minutes (block 24 still
    skipped, block 31 attention similarly)
  - SAE encode on GPU: < 1 s
  - one autoregressive log-prob pass: dominated by the forward itself
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import textwrap
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, "/home/gabriel/evo2-env/lib/python3.12/site-packages")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GPU_LAYERS = 19
STOP_AFTER_LAYER = 26
MAX_SEQ_LEN = 2000
DEFAULT_SEQ_LEN = 720
K = 64
SAE_PATH = (
    "/home/gabriel/.cache/huggingface/hub/models--Goodfire--Evo-2-Layer-26-Mixed/"
    "snapshots/a02b08a876b112d1c5da172e57a59e2bc76b1d70/"
    "sae-layer26-mixed-expansion_8-k_64.pt"
)
TARGET_LAYER = "blocks.26.mlp.l3"  # matches vast.ai LAYER (canonical SAE target)

EVO2_REPO = "evo2_7b"  # 1M context, FP8 projections, matches vast.ai
EVO2_FALLBACK_NO_FP8 = True  # if TE not installed, disable FP8 in cfg at load time
LAYER_PATCH_SENTINEL = "v2-mlp_l3"  # bumped when layer pin changed


# ---------------------------------------------------------------------------
# Idempotent vortex patches (applied at import time)
# ---------------------------------------------------------------------------

def _patch_file(path: str, old: str, new: str, label: str) -> bool:
    with open(path) as f:
        src = f.read()
    if old in src and new not in src:
        with open(path, "w") as f:
            f.write(src.replace(old, new, 1))
        print(f"[evo2_backend:patch:{label}] applied")
        return True
    return False


_model_py = "/home/gabriel/evo2-env/lib/python3.12/site-packages/vortex/model/model.py"
with open(_model_py) as _f:
    _model_src = _f.read()
if "import contextlib" not in _model_src:
    _lines = _model_src.splitlines(keepends=True)
    _last = 0
    for _i, _l in enumerate(_lines[:40]):
        if _l.startswith("import ") or _l.startswith("from "):
            _last = _i
    _lines.insert(_last + 1, "import contextlib\n")
    with open(_model_py, "w") as _f:
        _f.write("".join(_lines))

_patch_file(
    _model_py,
    "        with torch.cuda.device(x.device):\n"
    "            projected = self.projections(normalized)",
    "        projected = self.projections(normalized)  # PATCHED: cpu-safe",
    "model.proj_norm",
)
_old_init_block = (
    "            with torch.device(device):\n"
    "                # TELinear uses `device=\"cuda\"` device to allocate empty bias\n"
    "                # tensor. This makes sure that the empty tensor is allocated on the\n"
    "                # correct device. (torch.device(), unlike torch.cuda.device(),\n"
    "                # doesn't override current CUDA device.)\n"
    "                with torch.cuda.device(device):\n"
    "                    block = get_block(config, layer_idx, flash_fft=self.flash_fft)\n"
    "                    move_to_device(block, device)"
)
_new_init_block = (
    "            with torch.device(device):\n"
    "                if device.startswith('cuda'):\n"
    "                    _init_ctx = torch.cuda.device(device)\n"
    "                else:\n"
    "                    _init_ctx = contextlib.nullcontext()\n"
    "                with _init_ctx:\n"
    "                    block = get_block(config, layer_idx, flash_fft=self.flash_fft)\n"
    "                    move_to_device(block, device)  # PATCHED: cpu-safe"
)
_patch_file(_model_py, _old_init_block, _new_init_block, "model.__init__")


# ---------------------------------------------------------------------------
# Monkey-patch StripedHyena.__init__ to put blocks on cuda:0/cpu
# ---------------------------------------------------------------------------

import vortex.model.model as vmm  # noqa: E402

# Override CONFIG_MAP so Evo2("evo2_7b") loads the no-FP8 local config.
# RTX 3060 (sm_8.6) cannot run FP8 (needs sm_8.9+ on Hopper/Ada).
# The 1M model with FP8 disabled still matches vast.ai better than
# evo2_7b_base (different inner_mlp_size, rotary scaling, etc.).
import evo2.utils as _evo2_utils
if "configs/evo2-7b-1m.no-fp8.yml" not in _evo2_utils.CONFIG_MAP.get("evo2_7b", ""):
    import os as _os, shutil as _shutil
    _no_fp8_cfg = _os.path.join(_os.path.dirname(_evo2_utils.__file__), "configs", "evo2-7b-1m.no-fp8.yml")
    if not _os.path.exists(_no_fp8_cfg):
        # Generate it on the fly: copy evo2-7b-1m.yml and set use_fp8_input_projections=False
        import yaml as _yaml
        _src_cfg = _os.path.join(_os.path.dirname(_evo2_utils.__file__), "configs", "evo2-7b-1m.yml")
        _cfg = _yaml.safe_load(open(_src_cfg))
        _cfg["use_fp8_input_projections"] = False
        with open(_no_fp8_cfg, "w") as _f:
            _yaml.safe_dump(_cfg, _f)
    _evo2_utils.CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m.no-fp8.yml"
    print(f"[evo2_backend:patch:config] evo2_7b -> {_evo2_utils.CONFIG_MAP["evo2_7b"]} (FP8 disabled for sm_8.6)")


def _extract_init_source(cls):
    class_src = inspect.getsource(cls)
    lines = class_src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if "def __init__" in l)
    def_indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        cur = lines[end]
        if cur.strip() == "":
            end += 1
            continue
        if len(cur) - len(cur.lstrip()) <= def_indent and cur.strip():
            break
        end += 1
    return textwrap.dedent("".join(lines[start:end]))


_patched_init_src = _extract_init_source(vmm.StripedHyena)
_patched_init_src = _patched_init_src.replace(
    "super().__init__()", "torch.nn.Module.__init__(self)", 1
)
_old_dev_assign = (
    "        device_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)\n"
    "        device = f\"cuda:{device_idx}\" if torch.cuda.is_available() else \"cpu\""
)
_new_dev_assign = (
    f"        device_idx = 0  # PATCHED\n"
    f"        device = \"cuda:0\" if layer_idx < {GPU_LAYERS} else \"cpu\"  # PATCHED"
)
if _old_dev_assign not in _patched_init_src:
    raise SystemExit("Could not find device-assignment line in StripedHyena.__init__")
_patched_init_src = _patched_init_src.replace(_old_dev_assign, _new_dev_assign)
_globals = dict(vmm.__dict__)
exec(compile(_patched_init_src, "<patched_init>", "exec"), _globals)
vmm.StripedHyena.__init__ = _globals["__init__"]
print(f"[evo2_backend:patch:stripedhyena] first {GPU_LAYERS}/32 blocks -> cuda:0, rest -> cpu")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    sequence: str
    length: int
    per_position_logp: np.ndarray     # [L], per-position autoregressive log p
    mean_logp: float
    total_logp: float
    pseudo_perplexity: float
    per_position_entropy: np.ndarray  # [L]
    forward_seconds: float
    truncated_at_block: int            # which block we stopped at for this score


@dataclass
class VariantResult:
    ref_sequence: str
    alt_sequence: str
    ref_logp: float
    alt_logp: float
    delta_logp: float                  # alt - ref  (negative => deleterious)
    ref_per_pos: np.ndarray
    alt_per_pos: np.ndarray
    per_pos_delta: np.ndarray          # alt - ref at each position
    verdict: str                       # 'likely benign' / 'uncertain' / 'likely deleterious'
    forward_seconds: float


@dataclass
class FeatureResult:
    sequence: str
    length: int
    features: np.ndarray              # [L, 32768] sparse-ish, on CPU
    top_features: list                 # [(rank, feat_id, max_act, total_act, n_active)]
    per_position_count: np.ndarray     # [L] how many features fire at each position
    mean_nonzero_per_token: float
    sae_encode_seconds: float
    forward_seconds: float
    mean_embedding: np.ndarray = None  # PATCHED: block-26 mlp.l3 mean-pooled (4096,)


@dataclass
class EmbeddingResult:
    sequence: str
    length: int
    embeddings: np.ndarray            # [L, 4096] float32, block-26 hidden states
    mean_embedding: np.ndarray        # [4096] mean-pooled across positions
    forward_seconds: float


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Evo2Backend:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sae_W: Optional[torch.Tensor] = None
        self.sae_b_enc: Optional[torch.Tensor] = None
        self.sae_b_dec: Optional[torch.Tensor] = None
        self.K = K
        self.loaded = False
        self.loading = False
        self._lock = threading.Lock()
        self._last_status = "unloaded"

    # -------- lifecycle --------

    def ensure_loaded(self) -> str:
        if self.loaded:
            return "ready"
        with self._lock:
            if self.loaded:
                return "ready"
            self.loading = True
            try:
                self._load()
                self.loaded = True
                self._last_status = "ready"
            finally:
                self.loading = False
        return "ready"

    def _load(self):
        import torch as _torch
        _torch.set_grad_enabled(False)
        from evo2 import Evo2
        print(f"[evo2_backend] loading {EVO2_REPO} (1M context, 19/32 blocks on GPU via per-block weight streaming)…")
        t0 = time.time()
        self.model = Evo2(EVO2_REPO)
        self.tokenizer = self.model.tokenizer
        self._install_truncated_forward()
        print(f"[evo2_backend]   model loaded in {time.time()-t0:.0f}s")

        print(f"[evo2_backend] loading Goodfire Layer-26 SAE from {SAE_PATH}…")
        sd = torch.load(SAE_PATH, map_location="cuda:0", weights_only=False)
        self.sae_W = sd["_orig_mod.W"].to("cuda:0", dtype=torch.bfloat16)
        self.sae_b_enc = sd["_orig_mod.b_enc"].to("cuda:0", dtype=torch.bfloat16)
        self.sae_b_dec = sd["_orig_mod.b_dec"].to("cuda:0", dtype=torch.bfloat16)
        print(f"[evo2_backend]   SAE loaded: W {tuple(self.sae_W.shape)} K={self.K}")

    def status(self) -> dict:
        info = {
            "loaded": self.loaded,
            "loading": self.loading,
            "last_status": self._last_status,
        }
        if self.loaded:
            try:
                info["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
                info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            except Exception:
                pass
        return info

    # -------- internals --------

    def _block_dev(self, block) -> torch.device:
        p = next(block.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def _install_truncated_forward(self):
        """Replace the model's stateless_forward with a device-aware loop that
        handles CPU/GPU transitions and skips CPU attention blocks (which need
        Triton on CUDA). Goes through all blocks."""
        import types
        blocks = self.model.model.blocks
        # Cache for speed; recomputed inside the loop for accuracy
        def _forward(self, x, inference_params_dict=None, padding_mask=None):
            # Note: forward() already calls embedding_layer before stateless_forward
            gpu = torch.device("cuda:0")
            for block_idx, block in enumerate(blocks):
                # Per-block weight streaming (see _forward_to_block26)
                home_dev = next(block.parameters()).device
                if home_dev.type == "cpu":
                    block.to(gpu)
                    compute_dev = gpu
                else:
                    compute_dev = home_dev
                if x.device != compute_dev:
                    x = x.to(compute_dev)
                if compute_dev.type == "cpu" and getattr(block, "inner_mha_cls", None) is not None:
                    if home_dev.type == "cpu":
                        block.to(home_dev)
                    continue  # skip CPU attention blocks (Triton-only)
                x, _ = block(x, inference_params=None, padding_mask=padding_mask)
                if home_dev.type == "cpu":
                    block.to(home_dev)
            return x, None
        self.model.model.stateless_forward = types.MethodType(_forward, self.model.model)

    def _tokenize(self, seq: str) -> Tuple[torch.Tensor, int, str]:
        seq = _clean(seq)
        if not seq:
            raise ValueError("Empty sequence")
        warning = ""
        ids = self.tokenizer.tokenize(seq)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
            warning = f"Truncated to {MAX_SEQ_LEN} tokens."
        input_ids = torch.tensor(ids, dtype=torch.int64).unsqueeze(0)
        return input_ids, len(ids), warning

    def _forward_to_block26(self, seq: str, progress=None) -> Tuple[torch.Tensor, float]:
        """Returns (h: [1, L, 4096] on CPU, forward_seconds).
        Manually loops to block 26 so we don't rely on stateless_forward stopping."""
        self.ensure_loaded()
        input_ids, L, warn = self._tokenize(seq)
        emb_dev = next(self.model.model.embedding_layer.parameters()).device
        if progress is not None:
            progress(0.05, desc=f"Forward (L={L})…")
        t0 = time.time()
        with torch.no_grad():
          x = self.model.model.embedding_layer(input_ids.to(emb_dev))
          gpu = torch.device("cuda:0")
          for block_idx, block in enumerate(self.model.model.blocks):
              # Per-block weight streaming: if block lives on CPU, temporarily
              # move it to GPU for the matmul, then move it back. This avoids
              # single-threaded CPU matmuls while keeping peak VRAM low
              # (only one block's ~400MB at a time).
              home_dev = next(block.parameters()).device
              if home_dev.type == "cpu":
                  block.to(gpu)
                  compute_dev = gpu
              else:
                  compute_dev = home_dev
              if x.device != compute_dev:
                  x = x.to(compute_dev)
              # Skip CPU attention blocks (Triton rotary kernel is CUDA-only)
              if compute_dev.type == "cpu" and getattr(block, "inner_mha_cls", None) is not None:
                  if home_dev.type == "cpu":
                      block.to(home_dev)
                  if block_idx == STOP_AFTER_LAYER:
                      break
                  continue
              x, _ = block(x, inference_params=None, padding_mask=None)
              # Stream block back to its home device
              if home_dev.type == "cpu":
                  block.to(home_dev)
              if block_idx == STOP_AFTER_LAYER:
                  break
          h = x.detach().clone()  # break inference-tensor flag
        if progress is not None:
            progress(0.95, desc="Forward done")
        return h, time.time() - t0

    def _full_forward_logits(self, seq: str, progress=None) -> Tuple[torch.Tensor, float]:
        """Run the full model (with norm + unembed) and return logits [1, L, 512]."""
        self.ensure_loaded()
        input_ids, L, warn = self._tokenize(seq)
        emb_dev = next(self.model.model.embedding_layer.parameters()).device
        if progress is not None:
            progress(0.05, desc=f"Full forward (L={L})…")
        t0 = time.time()
        with torch.inference_mode():
            # Evo2 wrapper handles norm + unembed; calling model(input_ids)
            # returns the logits.  StripedHyena.forward returns (x, inference_dict),
            # Evo2.forward returns (logits, None), so unpack carefully.
            result = self.model(input_ids.to(emb_dev))
            if isinstance(result, tuple):
                logits = result[0]
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                logits = result
        if progress is not None:
            progress(0.95, desc="Full forward done")
        return logits, time.time() - t0

    # -------- public API --------

    def get_sae_features(self, h: torch.Tensor) -> Tuple[np.ndarray, float]:
        """h: [1, L, 4096]. Returns (features [L, 32768] float32, encode_seconds)."""
        self.ensure_loaded()
        t0 = time.time()
        with torch.no_grad():
            h_gpu = h.to("cuda:0", dtype=torch.bfloat16)
            x = h_gpu.float() - self.sae_b_dec.float()
            pre = x @ self.sae_W.float() + self.sae_b_enc.float()
            topk_vals, topk_idx = pre.topk(self.K, dim=-1)
            f = torch.zeros_like(pre)
            f.scatter_(-1, topk_idx, topk_vals.relu())
        return f[0].float().cpu().numpy(), time.time() - t0

    def autoregressive_logp(self, logits: torch.Tensor, input_ids: torch.Tensor) -> np.ndarray:
        """logits: [1, L, V], input_ids: [1, L]. Returns per-position log p of
        the *next* token (length L-1; last position dropped because there's
        no next token). This is the standard left-to-right language-model
        scoring used by variant-effect work.
        """
        with torch.inference_mode():
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)  # [L, V]
            target = input_ids[0]                                     # [L]
            # log p(x_{i+1} | x_<=i) at position i
            per_pos = log_probs[:-1, :].gather(1, target[1:].unsqueeze(1)).squeeze(1)
        return per_pos.cpu().numpy()

    def score_sequence(self, seq: str, progress=None) -> ScoreResult:
        """Per-position autoregressive log p + conservation. Runs the full
        model (through the unembed) to get logits.
        """
        self.ensure_loaded()
        input_ids, L, warn = self._tokenize(seq)
        logits, secs = self._full_forward_logits(seq, progress=progress)
        per_pos = self.autoregressive_logp(logits.cpu(), input_ids)
        # also compute per-position entropy of the predictive distribution
        with torch.inference_mode():
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            ent = -(log_probs.exp() * log_probs).sum(-1)
        per_pos_ent = ent.cpu().numpy()

        # Pad first position with the model's predicted log p (which is just
        # the marginal of x_1 from the BOS context). Set to 0 as NaN.
        per_pos = np.concatenate([[0.0], per_pos])
        per_pos_ent = per_pos_ent  # length L, includes position 0

        mean_lp = float(per_pos[1:].mean())
        total_lp = float(per_pos[1:].sum())
        ppl = float(math.exp(-mean_lp)) if mean_lp < 0 else float("inf")

        return ScoreResult(
            sequence=seq,
            length=L,
            per_position_logp=per_pos,
            mean_logp=mean_lp,
            total_logp=total_lp,
            pseudo_perplexity=ppl,
            per_position_entropy=per_pos_ent,
            forward_seconds=secs,
            truncated_at_block=32,
        )

    def variant_effect(self, ref: str, alt: str, progress=None) -> VariantResult:
        """Score two sequences (ref and alt). Return Δ log p. Both sequences
        must be the same length. Two forward passes, ~2x the per-seq cost.
        """
        self.ensure_loaded()
        ref_ids, Lr, wr = self._tokenize(ref)
        alt_ids, La, wa = self._tokenize(alt)
        if Lr != La:
            raise ValueError(f"ref/alt length mismatch ({Lr} vs {La}); must be equal")

        if progress is not None:
            progress(0.0, desc="Forward (ref)…")
        ref_logits, ref_secs = self._full_forward_logits(ref, progress=None)
        ref_per = self.autoregressive_logp(ref_logits.cpu(), ref_ids)
        if progress is not None:
            progress(0.5, desc="Forward (alt)…")
        alt_logits, alt_secs = self._full_forward_logits(alt, progress=None)
        alt_per = self.autoregressive_logp(alt_logits.cpu(), alt_ids)
        if progress is not None:
            progress(0.99, desc="Done")

        ref_logp = float(ref_per.sum())
        alt_logp = float(alt_per.sum())
        delta = alt_logp - ref_logp

        # Pad leading zero to align with the sequence's position indices
        ref_pad = np.concatenate([[0.0], ref_per])
        alt_pad = np.concatenate([[0.0], alt_per])

        # Verdict: -0.5 is a coarse threshold; tune per use case
        if delta < -0.5:
            verdict = "Likely deleterious (Δ < -0.5)"
        elif delta < -0.1:
            verdict = "Uncertain (Δ between -0.5 and -0.1)"
        else:
            verdict = "Likely benign (Δ > -0.1)"

        return VariantResult(
            ref_sequence=ref,
            alt_sequence=alt,
            ref_logp=ref_logp,
            alt_logp=alt_logp,
            delta_logp=delta,
            ref_per_pos=ref_pad,
            alt_per_pos=alt_pad,
            per_pos_delta=alt_pad - ref_pad,
            verdict=verdict,
            forward_seconds=ref_secs + alt_secs,
        )

    def extract_sae_features(self, seq: str, top_k: int = 20,
                             layer_name: str = TARGET_LAYER,
                             progress=None) -> FeatureResult:
        """Forward through the model, extract target-layer hidden states via
        the official return_embeddings API, encode with SAE.
        layer_name defaults to TARGET_LAYER = "blocks.26.mlp.l3" (vast.ai canonical)."""
        self.ensure_loaded()
        # Use official return_embeddings API for target layer h
        emb_res = self.extract_embeddings(seq, layer_name=layer_name, progress=progress)
        h = torch.from_numpy(emb_res.embeddings).unsqueeze(0)  # [1, L, 4096]
        fwd_secs = emb_res.forward_seconds
        if progress is not None:
            progress(0.95, desc="Encoding SAE…")
        feats, enc_secs = self.get_sae_features(h)
        if progress is not None:
            progress(0.99, desc="Done")

        flat = feats  # [L, 32768]
        counts = (flat > 0).sum(axis=0)        # [32768]
        max_act = flat.max(axis=0)             # [32768]
        total_act = flat.sum(axis=0)           # [32768]
        per_pos_count = (flat > 0).sum(axis=-1) # [L]

        top = torch.topk(torch.from_numpy(max_act), k=min(top_k, 32768))
        top_list = []
        for rank, (act, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            top_list.append({
                "rank": rank,
                "feat_id": int(idx),
                "max_act": float(act),
                "total_act": float(total_act[idx]),
                "active_in": int(counts[idx]),
                "n_tokens": int(flat.shape[0]),
            })

        return FeatureResult(
            sequence=seq,
            length=int(flat.shape[0]),
            features=feats,
            top_features=top_list,
            per_position_count=per_pos_count,
            mean_nonzero_per_token=float(per_pos_count.mean()),
            sae_encode_seconds=enc_secs,
            forward_seconds=fwd_secs,
            mean_embedding=emb_res.mean_embedding.astype(np.float32),  # PATCHED
        )

    def extract_embeddings(self, seq: str, layer_name: str = TARGET_LAYER,
                            progress=None) -> EmbeddingResult:
        """Extract hidden state embeddings from any Evo2 layer using the
        official return_embeddings=True, layer_names=[...] API.

        Uses forward hooks (no manual block looping). Runs the full forward
        pass; CPU-offloaded blocks use per-block weight streaming.

        Args:
            seq: nucleotide sequence
            layer_name: any submodule name in evo2_model.model
                (e.g. 'blocks.26', 'blocks.28.mlp.l3', 'embedding_layer')
                Default 'blocks.26' matches the Goodfire SAE target.
        """
        self.ensure_loaded()
        input_ids, L, _ = self._tokenize(seq)
        if progress is not None:
            progress(0.05, desc=f"Forward to {layer_name} (L={L})…")
        t0 = time.time()
        # Official Evo2 API: return_embeddings=True + layer_names
        # uses register_forward_hook internally
        outputs, embeddings = self.model(
            input_ids.to(next(self.model.model.embedding_layer.parameters()).device),
            return_embeddings=True,
            layer_names=[layer_name],
        )
        fwd_secs = time.time() - t0
        if progress is not None:
            progress(0.95, desc="Forward done")
        # embeddings[layer_name] is the output tensor from the hook
        h = embeddings[layer_name]
        if isinstance(h, tuple):
            h = h[0]
        h_np = h[0].float().cpu().numpy()  # [L, dim]
        mean_emb = h_np.mean(axis=0)       # [dim]
        return EmbeddingResult(
            sequence=seq,
            length=int(h_np.shape[0]),
            embeddings=h_np,
            mean_embedding=mean_emb,
            forward_seconds=fwd_secs,
        )

    # -------- new: batch variant effect --------

    def batch_variant_effect(self, ref: str, variants: list, progress=None) -> dict:
        """Score one reference and N alternates. Each variant is an "alt" string
        of the same length as the ref. Returns a list of per-variant dicts.
        Cost: (N+1) forward passes.
        """
        self.ensure_loaded()
        ref_c = _clean(ref)
        ref_ids, Lr, _ = self._tokenize(ref_c)
        ref_logits, _ = self._full_forward_logits(ref_c)
        ref_per = self.autoregressive_logp(ref_logits.cpu(), ref_ids)
        ref_logp = float(ref_per.sum())
        ref_per_pad = np.concatenate([[0.0], ref_per])
        ref_base = ref_c

        results = []
        n = len(variants)
        for i, alt in enumerate(variants):
            alt_c = _clean(alt)
            if not alt_c:
                results.append({"ref": ref_base, "alt": alt, "error": "empty after clean"})
                continue
            if len(alt_c) != len(ref_base):
                results.append({"ref": ref_base[:30] + "…", "alt": alt_c[:30] + "…",
                                 "error": f"length mismatch ({len(alt_c)} vs {len(ref_base)})"})
                continue
            var_pos = next((j + 1 for j, (a, b) in enumerate(zip(ref_base, alt_c)) if a != b), None)
            try:
                alt_ids, La, _ = self._tokenize(alt_c)
                alt_logits, _ = self._full_forward_logits(alt_c)
                alt_per = self.autoregressive_logp(alt_logits.cpu(), alt_ids)
                alt_logp = float(alt_per.sum())
                alt_per_pad = np.concatenate([[0.0], alt_per])
                delta = alt_logp - ref_logp
                if delta < -0.5:
                    verdict = "likely deleterious"
                elif delta < -0.1:
                    verdict = "uncertain"
                else:
                    verdict = "likely benign"
                results.append({
                    "alt": alt_c,
                    "pos": var_pos,
                    "ref_base": ref_base[var_pos-1] if var_pos else "—",
                    "alt_base": alt_c[var_pos-1] if var_pos else "—",
                    "ref_logp": ref_logp,
                    "alt_logp": alt_logp,
                    "delta_logp": delta,
                    "verdict": verdict,
                })
            except Exception as e:
                results.append({"alt": alt_c, "error": str(e)})
            if progress is not None:
                progress((i + 1) / max(1, n), desc=f"Variant {i+1}/{n}")
        return {"ref": ref_base, "results": results}

    # -------- new: compare two sequences --------

    def compare_sequences(self, seq1: str, seq2: str, progress=None) -> dict:
        """Score two sequences and return per-position log p for each, Δ, and
        features that differentiate them (fire on one but not the other).
        """
        self.ensure_loaded()
        s1 = _clean(seq1)
        s2 = _clean(seq2)
        if not s1 or not s2:
            return {"error": "empty sequence(s)"}

        if progress is not None:
            progress(0.0, desc="Scoring seq1…")
        h1, t1 = self._forward_to_block26(s1)
        feats1, _ = self.get_sae_features(h1)
        logp1, _ = self._full_forward_logits_scores(s1)

        if progress is not None:
            progress(0.5, desc="Scoring seq2…")
        h2, t2 = self._forward_to_block26(s2)
        feats2, _ = self.get_sae_features(h2)
        logp2, _ = self._full_forward_logits_scores(s2)

        if progress is not None:
            progress(0.99, desc="Done")

        # Truncate to the same length for comparison
        L = min(len(s1), len(s2))
        feats1 = feats1[:L]
        feats2 = feats2[:L]
        logp1 = logp1[:L]
        logp2 = logp2[:L]
        delta_logp = logp1 - logp2

        # Differentiating features: fire strongly on one but not the other
        max_act1 = feats1.max(0)
        max_act2 = feats2.max(0)
        score = max_act1 - max_act2
        top_s1 = np.argsort(-score)[:20]
        top_s2 = np.argsort(score)[:20]

        return {
            "seq1": s1[:L],
            "seq2": s2[:L],
            "length": L,
            "logp1": logp1,
            "logp2": logp2,
            "delta_logp": delta_logp,
            "feats1": feats1,
            "feats2": feats2,
            "features_only_in_seq1": [
                {"feat_id": int(i), "max_act_seq1": float(max_act1[i]),
                 "max_act_seq2": float(max_act2[i])}
                for i in top_s1 if max_act1[i] > max_act2[i]
            ],
            "features_only_in_seq2": [
                {"feat_id": int(i), "max_act_seq1": float(max_act1[i]),
                 "max_act_seq2": float(max_act2[i])}
                for i in top_s2 if max_act2[i] > max_act1[i]
            ],
        }

    def _full_forward_logits_scores(self, seq: str):
        """Returns (per_position_logp, seconds)."""
        ids, L, _ = self._tokenize(seq)
        logits, secs = self._full_forward_logits(seq)
        per = self.autoregressive_logp(logits.cpu(), ids)
        # Pad to length L
        per = np.concatenate([[0.0], per])
        return per, secs

    # -------- new: feature / position lookup --------

    @staticmethod
    def feature_positions(features: np.ndarray, feat_id: int) -> dict:
        """Given features [L, 32768] and a feature ID, return all positions
        where it fires and their activations."""
        if features is None or features.size == 0:
            return {"positions": [], "activations": []}
        if feat_id < 0 or feat_id >= features.shape[1]:
            return {"positions": [], "activations": []}
        col = features[:, feat_id]
        mask = col > 0
        positions = np.where(mask)[0].tolist()
        activations = col[mask].tolist()
        return {"positions": positions, "activations": activations}

    @staticmethod
    def top_features_at_position(features: np.ndarray, position: int, top_k: int = 10) -> list:
        """At a given position, return the top-k firing features."""
        if features is None or features.shape[0] == 0:
            return []
        if position < 0 or position >= features.shape[0]:
            return []
        row = features[position]
        active = np.where(row > 0)[0]
        if len(active) == 0:
            return []
        order = active[np.argsort(-row[active])][:top_k]
        return [{"feat_id": int(i), "activation": float(row[i])} for i in order]

    @staticmethod
    def low_likelihood_windows(per_position_logp: np.ndarray, window: int = 50,
                                z_threshold: float = -2.0) -> list:
        """Sliding-window z-score over per-position log p. Returns windows
        where the mean log p is `z_threshold` std below the global mean.
        """
        x = np.asarray(per_position_logp, dtype=np.float64)
        if x.size < window:
            return []
        # Rolling mean
        kernel = np.ones(window) / window
        means = np.convolve(x, kernel, mode="same")
        global_mean = float(x.mean())
        global_std = float(x.std() + 1e-9)
        z_scores = (means - global_mean) / global_std
        # Find contiguous regions where z < threshold
        low = z_scores < z_threshold
        regions = []
        start = None
        for i, b in enumerate(low):
            if b and start is None:
                start = i
            elif not b and start is not None:
                regions.append((start, i - 1, float(z_scores[start:i].min())))
                start = None
        if start is not None:
            regions.append((start, len(low) - 1, float(z_scores[start:].min())))
        # Sort by min z-score (most extreme first)
        regions.sort(key=lambda r: r[2])
        return [{"start": int(s) + 1, "end": int(e) + 1, "length": int(e - s + 1),
                 "min_z": float(z)} for s, e, z in regions]

    @staticmethod
    def rolling_mean(x: np.ndarray, window: int = 50) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")

    # -------- new: generation (with optional feature steering) --------

    def generate(self, prompt: str, n_tokens: int = 100, temperature: float = 1.0,
                 top_p: float = 0.9, force_feature_id: Optional[int] = None,
                 force_feature_strength: float = 4.0,
                 progress=None) -> dict:
        """Autoregressive generation using Evo2.

        If `force_feature_id` is set, the corresponding SAE feature is added
        to the residual stream at block 26 (the SAE's training layer) at
        every step. This is a simplified version of the feature-steering
        approach from the Evo2 / Goodfire papers.

        Cost: 1 forward pass per generated token (so ~`n_tokens` × ~0.3s
        on GPU plus a single CPU-side hook per step). For n_tokens=50
        with feature steering, expect ~3–5 min.
        """
        self.ensure_loaded()
        prompt_c = _clean(prompt)
        if not prompt_c:
            return {"error": "empty prompt"}
        ids = self.tokenizer.tokenize(prompt_c)
        generated = list(ids)
        t0 = time.time()

        # If steering, register a hook on block 26 to add the feature direction
        hook_handle = None
        if force_feature_id is not None and self.sae_W is not None:
            # The SAE decoder is W.T (tied weights). The feature direction
            # in residual-stream space is W[:, f] (column of encoder, or
            # equivalently row of W.T = decoder). Adding this with positive
            # strength forces the feature ON.
            feat_dir = self.sae_W[:, int(force_feature_id)].detach().clone().float()  # [4096]
            def _steer_hook(_mod, _inp, out):
                # out is a tensor of shape [B, L, 4096]
                if isinstance(out, tuple):
                    out = out[0]
                return out + force_feature_strength * feat_dir.to(out.device, out.dtype)
            hook_handle = self.model.model.blocks[evo2_backend.STOP_AFTER_LAYER].register_forward_hook(_steer_hook)

        try:
            for step in range(n_tokens):
                inp = torch.tensor(generated, dtype=torch.int64).unsqueeze(0)
                emb_dev = next(self.model.model.embedding_layer.parameters()).device
                with torch.inference_mode():
                    logits = self.model(inp.to(emb_dev))
                # Sample next token from logits at the LAST position
                last = logits[0, -1].float() / max(1e-6, temperature)
                probs = torch.softmax(last, dim=-1)
                # Top-p
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= top_p
                keep[0] = True
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[sorted_idx] = keep
                probs = probs * mask.float()
                probs = probs / (probs.sum() + 1e-9)
                next_id = int(torch.multinomial(probs, 1).item())
                generated.append(next_id)
                if progress is not None and n_tokens > 0:
                    progress((step + 1) / n_tokens, desc=f"Gen {step+1}/{n_tokens}")
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Decode the generated tokens to a string
        # The tokenizer maps token IDs back to bytes/strings; we use the
        # same character-level scheme (vocab=512: A=65, C=67, G=71, T=84).
        # Build an inverse mapping.
        inv = {65: 'A', 67: 'C', 71: 'G', 84: 'T'}
        # Also try to get the tokenizer's own decode if it has one
        if hasattr(self.tokenizer, 'detokenize'):
            try:
                full_seq = self.tokenizer.detokenize(generated)
            except Exception:
                full_seq = ''.join(inv.get(int(t), '?') for t in generated)
        else:
            full_seq = ''.join(inv.get(int(t), '?') for t in generated)

        # Try to extract just the generated part (after prompt length)
        # The tokenizer's tokenize function may prepend BOS, so we use
        # the prompt length in characters as a heuristic.
        prompt_len = len(prompt_c)
        gen_seq = full_seq[prompt_len:] if len(full_seq) > prompt_len else full_seq

        return {
            "prompt": prompt_c,
            "generated_sequence": gen_seq,
            "full_sequence": full_seq,
            "n_tokens_generated": n_tokens,
            "elapsed_seconds": time.time() - t0,
            "forced_feature_id": force_feature_id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import math


def _clean(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith(">"):
        out = []
        for line in s.splitlines():
            if line.startswith(">"):
                continue
            out.append(line.strip())
        s = "".join(out)
    s = "".join(s.split()).upper()
    if s.endswith("*"):
        s = s[:-1]
    # Strip U (uracil) -> T for DNA
    s = s.replace("U", "T")
    return s
