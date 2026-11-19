"""Trains three small byte-level BPE tokenizers that mirror real model families.

Real tokenizers cannot be downloaded in every environment, so the tests build
look-alikes with the same pre-tokenizer regexes and options:

  tok-qwen2   Split(Qwen2 regex, isolated) + ByteLevel(use_regex=False), ChatML specials
  tok-llama3  Split(Llama-3 regex) + ByteLevel, ignore_merges, <|begin_of_text|> BOS
  tok-gpt2    ByteLevel(use_regex=True) (the classic GPT-2 setup)

Usage: python tests/make_test_tokenizers.py <out_dir>
"""
import glob
import json
import os
import sys

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+|\s+(?!\S)|\s+")
LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+")

CHATML_TEMPLATE = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']"
                   " + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}"
                   "{{ '<|im_start|>assistant\\n' }}{% endif %}")
LLAMA3_TEMPLATE = ("{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + "
                   "'<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}"
                   "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                   "{% endif %}")


def corpus():
    """English-ish text from the Python stdlib plus some multilingual lines."""
    texts = []
    stdlib = os.path.dirname(os.__file__)
    for path in sorted(glob.glob(os.path.join(stdlib, "*.py")))[:80]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    texts += [
        "你好，世界！今天天气很好。我们一起去公园散步吧。",
        "Café, naïve, résumé, coöperate — Ünïcödé ßtraße.",
        "Emoji: 😀 👍🏽 🚀✨ and numbers 1234567890 3.14159 2026-11-18.",
        "Привет, мир! Это тест токенизатора.",
        "こんにちは世界。トークナイザーのテストです。",
    ] * 20
    return texts


def build(kind):
    if kind == "qwen2":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    elif kind == "llama3":
        tok = Tokenizer(models.BPE(ignore_merges=True))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(LLAMA3_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                    "<|eot_id|>"]
    elif kind == "gpt2":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        specials = ["<|endoftext|>"]
    else:
        raise ValueError(kind)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=2000, special_tokens=specials, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus(), trainer)
    if kind == "llama3":
        bos = tok.token_to_id("<|begin_of_text|>")
        tok.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(single="<|begin_of_text|> $A", pair="<|begin_of_text|> $A $B:1",
                                          special_tokens=[("<|begin_of_text|>", bos)]),
        ])
    return tok


def save(kind, out_dir):
    d = os.path.join(out_dir, f"tok-{kind}")
    os.makedirs(d, exist_ok=True)
    tok = build(kind)
    tok.save(os.path.join(d, "tokenizer.json"))
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast"}
    if kind == "qwen2":
        cfg.update(chat_template=CHATML_TEMPLATE, eos_token="<|im_end|>")
    elif kind == "llama3":
        cfg.update(chat_template=LLAMA3_TEMPLATE, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return d


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/data"
    for k in ("qwen2", "llama3", "gpt2"):
        print(save(k, out))
