"""Model parity: C++ paged forward pass vs Hugging Face transformers.

The C++ side runs the prompt as chunked prefill (several chunks, so later
chunks attend to a cached prefix) followed by token-by-token decode, with a
small block size so sequences span many non-contiguous blocks. Two copies of
the sequence run in the same batch with different block tables. Logits for
every position must match HF's single full-sequence forward pass.

Usage: python tests/test_model_parity.py <build_dir> <model_dir> [<model_dir> ...]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def check(build_dir, model_dir, device="cpu", length=45, prefill=20, chunk=7, block_size=4):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, model.config.vocab_size, (length,), generator=g)
    with torch.no_grad():
        ref = model(ids.unsqueeze(0)).logits[0].numpy()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "logits.bin")
        cmd = [os.path.join(build_dir, "pagedserve-dev"), "logits", "--model", model_dir, "--tokens",
               ",".join(map(str, ids.tolist())), "--prefill", str(prefill), "--chunk", str(chunk),
               "--block-size", str(block_size), "--out", out, "--device", device]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            return False
        meta = json.loads(res.stdout.strip().splitlines()[-1])
        got = np.fromfile(out, dtype=np.float32).reshape(meta["rows"], meta["vocab"])
    err = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).max())
    argmax_ok = bool((got.argmax(-1) == ref.argmax(-1)).all())
    ok = err < 1e-3 * max(1.0, scale) and argmax_ok and meta["max_dup_diff"] < 1e-4
    print(f"{'OK  ' if ok else 'FAIL'} {model_dir} [{device}]: max|diff|={err:.2e} (logit scale {scale:.1f}), "
          f"argmax match={argmax_ok}, batch-duplicate diff={meta['max_dup_diff']:.1e}")
    return ok


if __name__ == "__main__":
    build_dir = sys.argv[1]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = [check(build_dir, d, dev) for d in sys.argv[2:] for dev in devices]
    sys.exit(0 if all(results) else 1)
