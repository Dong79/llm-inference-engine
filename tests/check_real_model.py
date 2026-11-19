"""Real-checkpoint check: pagedserve's greedy output vs Hugging Face transformers.

Starts the server on a real model (e.g. Qwen2.5-0.5B-Instruct) and, for a few
chat prompts:
  1. builds the prompt token ids with HF's own chat template,
  2. runs greedy decoding in transformers (manual loop, no generation_config
     sampling or repetition penalty),
  3. sends the same token ids to /v1/completions with temperature 0,
  4. sends the same messages to /v1/chat/completions (checks our built-in chat
     template produces the same prompt as HF's Jinja template).

In float32 the texts should match exactly. In bfloat16 they usually agree for
a while and may then diverge: different kernels round differently and greedy
decoding amplifies a near-tie. That is expected, not a bug.

Usage: python tests/check_real_model.py <build_dir> <model_dir> [--dtype float32] [--max-tokens 64]
"""
import argparse
import http.client
import json
import os
import socket
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a short poem about GPUs.",
    "What is 17 * 23? Show your work.",
    "List three uses of a hash table.",
    "用一句话介绍你自己。",
]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post(port, path, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    if r.status != 200:
        raise RuntimeError(data)
    return data


def hf_greedy(model, ids, max_tokens, eos):
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        res = model(torch.tensor([ids], device=dev), use_cache=True)
        for _ in range(max_tokens):
            nxt = int(res.logits[0, -1].argmax())
            if nxt in eos:
                break
            out.append(nxt)
            res = model(torch.tensor([[nxt]], device=dev), past_key_values=res.past_key_values, use_cache=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("model_dir")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    port = free_port()
    log = open(os.path.join(args.build_dir, "check_real_model.log"), "w")
    proc = subprocess.Popen([os.path.join(args.build_dir, "pagedserve"), "--model", args.model_dir, "--port",
                             str(port), "--host", "127.0.0.1", "--dtype", args.dtype, "--max-model-len", "4096",
                             "--gpu-memory-utilization", "0.5"], stdout=log, stderr=subprocess.STDOUT)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=getattr(torch, args.dtype)).to(dev).eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        if tok.eos_token_id is not None:
            eos.add(tok.eos_token_id)
        for _ in range(600):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start; see " + log.name)

        exact = 0
        for q in PROMPTS:
            msgs = [SYSTEM, {"role": "user", "content": q}]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text, add_special_tokens=False).input_ids
            ref = tok.decode(hf_greedy(model, ids, args.max_tokens, eos), skip_special_tokens=True)
            comp = post(port, "/v1/completions", {"prompt": ids, "max_tokens": args.max_tokens, "temperature": 0})
            chat = post(port, "/v1/chat/completions", {"messages": msgs, "max_tokens": args.max_tokens,
                                                       "temperature": 0})
            got = comp["choices"][0]["text"]
            got_chat = chat["choices"][0]["message"]["content"]
            same = got == ref
            exact += same
            common = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
            print(f"{'MATCH   ' if same else 'DIVERGES'} {q[:40]!r}: "
                  + ("" if same else f"first difference at char {common} of {len(ref)}; ")
                  + f"chat template {'agrees' if got_chat == got else 'DIFFERS'}")
            if not same:
                print(f"    hf : {ref[:200]!r}\n    ours: {got[:200]!r}")
        print(f"\n{exact}/{len(PROMPTS)} prompts identical to transformers ({args.dtype})")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
