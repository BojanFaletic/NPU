"""Hand-rolled SmolLM2-135M forward pass. Matches HF Llama token-for-token.

Arch: 30 x {RMSNorm -> GQA attn (9q/3kv, d=64, RoPE) -> RMSNorm -> SwiGLU MLP},
final RMSNorm, tied lm_head. No biases anywhere.
"""
from __future__ import annotations
import math, argparse, time
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


@dataclass
class Config:
    dim: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    ffn_dim: int = 1536
    vocab: int = 49152
    max_pos: int = 8192
    rope_theta: float = 100000.0  # SmolLM2 default; overridden from HF config
    rms_eps: float = 1e-5


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # compute in fp32 for stability (matches HF)
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rms).to(x.dtype) * w


def build_rope_cache(head_dim: int, max_pos: int, theta: float, dtype, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                       # [max_pos, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [max_pos, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [T, D]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# --- KV cache ------------------------------------------------------------
# A cache holds one (k, v) pair per layer, each of shape [B, Hkv, T_cached, Dh].
# `None` entries mean the layer hasn't been populated yet (empty cache).
KVCache = list[tuple[torch.Tensor, torch.Tensor] | None]


def empty_cache(cfg: Config) -> KVCache:
    return [None] * cfg.n_layers


class Layer:
    __slots__ = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down", "ln1", "ln2", "npu")

    def __init__(self, sd: dict, i: int):
        p = f"model.layers.{i}."
        self.wq = sd[p + "self_attn.q_proj.weight"]
        self.wk = sd[p + "self_attn.k_proj.weight"]
        self.wv = sd[p + "self_attn.v_proj.weight"]
        self.wo = sd[p + "self_attn.o_proj.weight"]
        self.w_gate = sd[p + "mlp.gate_proj.weight"]
        self.w_up = sd[p + "mlp.up_proj.weight"]
        self.w_down = sd[p + "mlp.down_proj.weight"]
        self.ln1 = sd[p + "input_layernorm.weight"]
        self.ln2 = sd[p + "post_attention_layernorm.weight"]
        self.npu: dict[str, object] | None = None  # filled by SmolLM when --npu

    def _lin(self, name: str, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """F.linear(x, W) — routed through NPU if this layer has an NpuLinear."""
        if self.npu is not None and name in self.npu:
            return self.npu[name](x)
        return F.linear(x, W)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cfg: Config,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        h = rms_norm(x, self.ln1, cfg.rms_eps)
        q = self._lin("wq", h, self.wq).view(B, T, Hq,  Dh).transpose(1, 2)
        k = self._lin("wk", h, self.wk).view(B, T, Hkv, Dh).transpose(1, 2)
        v = self._lin("wv", h, self.wv).view(B, T, Hkv, Dh).transpose(1, 2)
        # RoPE uses absolute positions [start_pos, start_pos + T)
        q = apply_rope(q, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])
        k = apply_rope(k, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])
        # append to cache
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        new_cache = (k, v)
        # GQA: repeat kv to match q heads (post-concat, so cache stays compact)
        rep = Hq // Hkv
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)
        Tk = k_exp.shape[2]
        # causal attention. For decode (T=1) no mask needed; for prefill, mask is causal.
        att = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(Dh)  # [B, Hq, T, Tk]
        if T > 1:
            # position i in q attends to positions [0, start_pos + i] in k
            # simplest: build a [T, Tk] mask
            row = torch.arange(T, device=x.device)[:, None] + start_pos
            col = torch.arange(Tk, device=x.device)[None, :]
            mask = torch.where(col <= row, 0.0, float("-inf")).to(att.dtype)
            att = att + mask
        # Softmax: prefill (T>1) goes to NPU if enabled, else CPU (fp32). Decode
        # (T=1) always stays on CPU — rows would be tiny and the xclbin cache
        # would mint a new one each step as Tk grows.
        if self.npu is not None and "softmax" in self.npu and T > 1:
            att = self.npu["softmax"](att).to(x.dtype)
        else:
            att = F.softmax(att.float(), dim=-1).to(x.dtype)
        o = torch.matmul(att, v_exp).transpose(1, 2).contiguous().view(B, T, Hq * Dh)
        x = x + self._lin("wo", o, self.wo)
        # --- MLP (SwiGLU) ---
        h = rms_norm(x, self.ln2, cfg.rms_eps)
        gate = F.silu(self._lin("w_gate", h, self.w_gate))
        up = self._lin("w_up", h, self.w_up)
        x = x + self._lin("w_down", gate * up, self.w_down)
        return x, new_cache


class SmolLM:
    def __init__(self, sd: dict, cfg: Config):
        self.cfg = cfg
        self.embed = sd["model.embed_tokens.weight"]
        self.layers = [Layer(sd, i) for i in range(cfg.n_layers)]
        self.final_norm = sd["model.norm.weight"]
        self.lm_head = self.embed  # tied
        dtype = self.embed.dtype
        device = self.embed.device
        self.cos, self.sin = build_rope_cache(cfg.head_dim, cfg.max_pos, cfg.rope_theta, dtype, device)

    def enable_npu(
        self,
        ops: tuple[str, ...] = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"),
        softmax: bool = True,
    ) -> None:
        """Route the named linear ops (and optionally softmax) through NPU on every layer.

        NpuLinear lazily compiles xclbins per (M, K, N) — first forward triggers
        a burst of Peano/aiecc calls (~1-2s each). Weights are cast to bf16.
        Softmax uses a single shared NpuSoftmax (no per-layer weights), keyed on
        (rows, L) shape.
        """
        from npu.linear import NpuLinear  # local import to keep CPU path zero-cost
        # One NpuLinear per (layer, op) since weights differ per layer.
        for layer in self.layers:
            layer.npu = {}
            for op in ops:
                W = getattr(layer, op)
                layer.npu[op] = NpuLinear(W)
        if softmax:
            from npu.softmax import NpuSoftmax
            shared_softmax = NpuSoftmax(n_cores=1)
            for layer in self.layers:
                layer.npu["softmax"] = shared_softmax

    def forward(
        self,
        ids: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache]:
        if cache is None:
            cache = empty_cache(self.cfg)
        x = self.embed[ids]
        new_cache: KVCache = [None] * self.cfg.n_layers
        for i, layer in enumerate(self.layers):
            x, new_cache[i] = layer.forward(x, self.cos, self.sin, self.cfg, cache[i], start_pos)
        x = rms_norm(x, self.final_norm, self.cfg.rms_eps)
        return F.linear(x, self.lm_head), new_cache

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self.forward(ids)
                next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
            return ids

        # prefill
        logits, cache = self.forward(ids, cache=None, start_pos=0)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        out = [ids, next_id]
        pos = ids.shape[1]
        # decode loop
        for _ in range(max_new_tokens - 1):
            logits, cache = self.forward(next_id, cache=cache, start_pos=pos)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(next_id)
            pos += 1
        return torch.cat(out, dim=1)


def load(dtype=torch.float32) -> tuple[SmolLM, object, object]:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    hf.eval()
    sd = {k: v.detach() for k, v in hf.state_dict().items()}
    cfg_hf = AutoConfig.from_pretrained(MODEL_ID)
    rope_theta = float(getattr(cfg_hf, "rope_parameters", {}).get("rope_theta",
                       getattr(cfg_hf, "rope_theta", 10000.0)))
    cfg = Config(rope_theta=rope_theta)
    return SmolLM(sd, cfg), tok, hf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--check", action="store_true", help="compare logits vs HF")
    p.add_argument("--no-cache", action="store_true", help="disable KV cache (slow)")
    p.add_argument("--compare-cache", action="store_true", help="compare cache vs no-cache outputs")
    p.add_argument("--npu", action="store_true",
                   help="route matmuls through NPU (first call compiles, subsequent cached)")
    args = p.parse_args()

    model, tok, hf = load(torch.float32)
    if args.npu:
        print("enabling NPU backend for all MLP + attention projections…")
        model.enable_npu()
    ids = tok(args.prompt, return_tensors="pt").input_ids

    if args.check:
        with torch.no_grad():
            hf_logits = hf(ids).logits
            my_logits, _ = model.forward(ids)
        diff = (hf_logits - my_logits).abs()
        print(f"logits max|Δ| = {diff.max().item():.3e}   mean|Δ| = {diff.mean().item():.3e}")
        hf_top = hf_logits[0, -1].argmax().item()
        my_top = my_logits[0, -1].argmax().item()
        print(f"top-1 token: hf={hf_top} ({tok.decode([hf_top])!r})  mine={my_top} ({tok.decode([my_top])!r})")

    if args.compare_cache:
        a = model.generate(ids, args.max_new_tokens, use_cache=False)
        b = model.generate(ids, args.max_new_tokens, use_cache=True)
        print(f"cache vs no-cache tokens match: {torch.equal(a, b)}")
        if not torch.equal(a, b):
            print(f"  no-cache: {tok.decode(a[0])!r}")
            print(f"  cache   : {tok.decode(b[0])!r}")

    t0 = time.time()
    out = model.generate(ids, args.max_new_tokens, use_cache=not args.no_cache)
    dt = time.time() - t0
    n = out.shape[1] - ids.shape[1]
    mode = "no-cache" if args.no_cache else "kv-cache"
    print(f"[{mode}] generated {n} tok in {dt:.2f}s ({n/dt:.1f} tok/s)")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
