"""Builds tiny random-weight checkpoints in Hugging Face format for tests.

  tiny-llama3  LlamaForCausalLM, GQA, Llama-3 RoPE scaling, untied lm_head, Llama-3 tokenizer
  tiny-qwen2   Qwen2ForCausalLM, GQA, QKV bias, tied embeddings, ChatML tokenizer
  tiny-qwen3   Qwen3ForCausalLM, QK-norm, head_dim (64) != hidden/heads (32), ChatML tokenizer

head_dim is 32 or 64 so that on a GPU the tests also exercise the CUDA kernel.

Weights use a large init std so logits are peaked and greedy decoding is
stable against float rounding; biases and norm weights are randomised so
every code path actually matters.

Usage: python tests/make_tiny_models.py <tokenizers_dir> <out_dir>
"""
import json
import os
import shutil
import sys

import torch
from tokenizers import Tokenizer
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM)


def randomise(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            elif "norm" in name:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.25)


def build(kind, tok_root, out_root):
    tok_dir = os.path.join(tok_root, "tok-llama3" if kind == "llama3" else "tok-qwen2")
    tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    common = dict(vocab_size=vocab, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, rms_norm_eps=1e-6)
    gen_eos = None
    if kind == "llama3":
        eos = [tok.token_to_id("<|end_of_text|>"), tok.token_to_id("<|eot_id|>")]
        cfg = LlamaConfig(**common, rope_theta=500000.0, tie_word_embeddings=False,
                          bos_token_id=tok.token_to_id("<|begin_of_text|>"), eos_token_id=eos,
                          rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                        "high_freq_factor": 4.0, "original_max_position_embeddings": 256})
        model = LlamaForCausalLM(cfg)
    elif kind == "qwen2":
        cfg = Qwen2Config(**common, rope_theta=1000000.0, tie_word_embeddings=True,
                          eos_token_id=tok.token_to_id("<|endoftext|>"))
        model = Qwen2ForCausalLM(cfg)
        gen_eos = [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]
    elif kind == "qwen3":
        cfg = Qwen3Config(**common, head_dim=64, rope_theta=1000000.0, tie_word_embeddings=True,
                          eos_token_id=tok.token_to_id("<|endoftext|>"))
        model = Qwen3ForCausalLM(cfg)
    else:
        raise ValueError(kind)
    randomise(model, seed={"llama3": 11, "qwen2": 22, "qwen3": 33}[kind])
    if cfg.tie_word_embeddings:
        model.tie_weights()
    model.eval()
    out = os.path.join(out_root, f"tiny-{kind}")
    model.save_pretrained(out, safe_serialization=True)
    if gen_eos is not None:
        with open(os.path.join(out, "generation_config.json"), "w") as f:
            json.dump({"eos_token_id": gen_eos}, f)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(tok_dir, name), os.path.join(out, name))
    return out


if __name__ == "__main__":
    tok_root, out_root = sys.argv[1], sys.argv[2]
    for k in ("llama3", "qwen2", "qwen3"):
        print(build(k, tok_root, out_root))
