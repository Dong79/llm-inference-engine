"""End-to-end engine correctness: greedy outputs must equal HF token-for-token.

One workload, several engine configurations. The workload is built to hit
every code path at once:
  * prompts sharing a long system prefix, and a "second turn" whose prompt
    extends an earlier prompt          -> prefix-cache hits
  * a long prompt with a small token budget -> chunked prefill
  * staggered arrivals                  -> admission while others decode
  * n=3 greedy sampling                 -> fork + copy-on-write of the last block
  * a tiny KV cache in one config        -> preemption and recompute

Usage: python tests/test_engine_parity.py <build_dir> <model_dir>
"""
import json
import os
import random
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM


def greedy_reference(model, prompt, max_tokens):
    ids = list(prompt)
    min_margin = float("inf")
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([ids])).logits[0, -1]
            top2 = torch.topk(logits, 2).values
            min_margin = min(min_margin, float(top2[0] - top2[1]))
            ids.append(int(logits.argmax()))
    return ids[len(prompt):], min_margin


def workload(vocab, block_size):
    rng = random.Random(7)
    tok = lambda n: [rng.randrange(vocab) for _ in range(n)]
    system = tok(5 * block_size + 3)  # shared prefix spanning several blocks
    reqs = []
    for i in range(6):
        reqs.append({"prompt_ids": system + tok(rng.randint(2, 9)), "max_tokens": rng.randint(4, 24),
                     "arrive_step": rng.randint(0, 12)})
    # Second turn of request 0: its prompt extends request 0's prompt.
    reqs.append({"prompt_ids": reqs[0]["prompt_ids"] + tok(6), "max_tokens": 10, "arrive_step": 40})
    # Long prompt: needs several chunks under a small token budget.
    reqs.append({"prompt_ids": tok(70), "max_tokens": 12, "arrive_step": 3})
    # Parallel sampling with greedy decoding: all 3 choices must be identical.
    reqs.append({"prompt_ids": tok(2 * block_size + 2), "max_tokens": 9, "n": 3, "arrive_step": 5})
    # Prompt that is exactly a whole number of blocks, repeated (full prefix hit).
    exact = tok(4 * block_size)
    reqs.append({"prompt_ids": exact, "max_tokens": 6, "arrive_step": 0})
    reqs.append({"prompt_ids": exact, "max_tokens": 6, "arrive_step": 30})
    # Single-token prompt and single-token output.
    reqs.append({"prompt_ids": tok(1), "max_tokens": 1, "arrive_step": 2})
    for r in reqs:
        r.update(temperature=0.0, ignore_eos=True)
    return reqs


CONFIGS = {
    "continuous+chunked+prefix": ["--max-num-batched-tokens", "16", "--num-kv-blocks", "400"],
    "tight-memory (preemption)": ["--max-num-batched-tokens", "32", "--num-kv-blocks", "34"],
    "static batching": ["--static-batching", "--max-num-seqs", "4", "--num-kv-blocks", "400"],
    "no prefix cache, no chunking": ["--no-prefix-caching", "--no-chunked-prefill", "--num-kv-blocks", "400",
                                      "--max-num-batched-tokens", "256"],
}


def main():
    build_dir, model_dir = sys.argv[1], sys.argv[2]
    block_size = 4
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    reqs = workload(model.config.vocab_size, block_size)
    refs = []
    worst_margin = float("inf")
    for r in reqs:
        out, margin = greedy_reference(model, r["prompt_ids"], r["max_tokens"])
        refs.append(out)
        worst_margin = min(worst_margin, margin)
    print(f"reference ready: {len(reqs)} requests, smallest top-1/top-2 logit margin {worst_margin:.3f}")
    printed_device = False

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        req_path = os.path.join(td, "reqs.json")
        with open(req_path, "w") as f:
            json.dump(reqs, f)
        for name, extra in CONFIGS.items():
            out_path = os.path.join(td, "out.json")
            cmd = [os.path.join(build_dir, "pagedserve-dev"), "offline", "--model", model_dir, "--requests", req_path,
                   "--out", out_path, "--block-size", str(block_size), "--max-model-len", "128", "--dtype", "float32"] + extra
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"FAIL {name}: exit {res.returncode}\n{res.stderr[-2000:]}")
                failures += 1
                continue
            data = json.load(open(out_path))
            if not printed_device:
                desc = [l for l in res.stderr.splitlines() if l.startswith(("device", "attention"))]
                print("  engine: " + " | ".join(" ".join(l.split()) for l in desc))
                printed_device = True
            s = data["summary"]
            bad = 0
            for i, (r, ref) in enumerate(zip(reqs, refs)):
                for c in data["results"][i]["choices"]:
                    if c["token_ids"] != ref:
                        bad += 1
                        if bad <= 2:
                            print(f"  req {i}: expected {ref}\n          got      {c['token_ids']}")
            checks = []
            if name.startswith("continuous"):
                checks.append(("prefix hits", s["prefix_hit_tokens"] > 0))
                checks.append(("copy-on-write", s["cow_copies"] > 0))
            if "preemption" in name:
                checks.append(("preemptions", s["preemptions"] > 0))
            if "no prefix" in name:
                checks.append(("no prefix hits", s["prefix_hit_tokens"] == 0))
            failed_checks = [c for c, ok in checks if not ok]
            ok = bad == 0 and not failed_checks
            failures += 0 if ok else 1
            print(f"{'OK  ' if ok else 'FAIL'} {name:30s} mismatches={bad} steps={s['engine_steps']} "
                  f"preemptions={s['preemptions']} cow={s['cow_copies']} "
                  f"prefix_hit={s['prefix_hit_tokens']}/{s['prefix_query_tokens']}"
                  + (f"  failed: {failed_checks}" if failed_checks else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
