"""HF reference: loads SmolLM2-135M and generates. Our correctness oracle."""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print(f"loading {MODEL_ID} ({args.dtype}) …")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s  |  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"\nprompt : {args.prompt!r}  (tokens={ids.shape[1]})")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    dt = time.time() - t0
    new_tokens = out.shape[1] - ids.shape[1]
    print(f"generated {new_tokens} tokens in {dt:.2f}s  ({new_tokens/dt:.1f} tok/s)")
    print(f"\noutput :\n{tok.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
