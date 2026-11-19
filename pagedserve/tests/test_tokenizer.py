"""Checks the C++ tokenizer against Hugging Face `tokenizers` token-for-token.

Usage: python tests/test_tokenizer.py <build_dir> <tokenizer_dir> [<tokenizer_dir> ...]
"""
import json
import os
import random
import subprocess
import sys
import unicodedata

from tokenizers import Tokenizer

FIXED = [
    "",
    "Hello, world!",
    "I'm here, you're there; they'LL see. DON'T stop. It's 'quoted'.",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1e10 0xFF",
    "Whitespace   runs\t\tand\n\nnewlines\r\n  indented\n    code",
    "trailing spaces   ",
    "   leading spaces",
    "你好，世界！今天天气很好。",
    "Emoji 😀👍🏽🚀 mixed with text✨.",
    "Café naïve résumé — Ünïcödé ßtraße",
    "Привет, мир! こんにちは世界。",
    "def f(x):\n    return x ** 2  # square\n",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>",
    "text<|endoftext|>more text<|endoftext|>",
    "!!!??? ... ---- ==== $$$ ### @@@",
    "a b c　d",  # non-ASCII whitespace
    "ﬁ ligature, combining é accent",
]


def random_strings(n, seed=0):
    rng = random.Random(seed)
    alphabet = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()[]{}<>\n\t-_=+*/\\"
                "你好世界éüßñ😀👍ДЖЯ")
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80))) for _ in range(n)]


def main():
    build_dir, tok_dirs = sys.argv[1], sys.argv[2:]
    exe = os.path.join(build_dir, "pagedserve-tokenize")
    strings = FIXED + random_strings(300)
    failures = 0
    for d in tok_dirs:
        hf = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        # The C++ tokenizer does not implement NFC normalization (used by Qwen);
        # skip inputs that NFC would change, and say so.
        with open(os.path.join(d, "tokenizer.json")) as f:
            normalizer = json.load(f).get("normalizer") or {}
        nfc = "NFC" in json.dumps(normalizer)
        cases = [s for s in strings if not (nfc and unicodedata.normalize("NFC", s) != s)]
        skipped = len(strings) - len(cases)
        res = subprocess.run([exe, d], input=json.dumps(cases), capture_output=True, text=True, check=True)
        out = json.loads(res.stdout)
        bad = 0
        for s, ids, dec in zip(cases, out["ids"], out["decoded"]):
            ref = hf.encode(s, add_special_tokens=False).ids
            ref_dec = hf.decode(ref, skip_special_tokens=True)
            if ids != ref or dec != ref_dec:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH in {d} for {s!r}\n    hf : {ref}\n    cpp: {ids}\n"
                          f"    hf dec : {ref_dec!r}\n    cpp dec: {dec!r}")
        bos_ref = hf.encode("x", add_special_tokens=True).ids
        bos_ref = bos_ref[0] if len(bos_ref) > len(hf.encode("x", add_special_tokens=False).ids) else None
        if out["bos"] != bos_ref:
            bad += 1
            print(f"  BOS mismatch in {d}: hf={bos_ref} cpp={out['bos']}")
        note = f" ({skipped} skipped: need NFC normalization)" if skipped else ""
        print(f"{'OK  ' if bad == 0 else 'FAIL'} {d}: {len(cases)} strings, {bad} mismatches{note}")
        failures += bad
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
