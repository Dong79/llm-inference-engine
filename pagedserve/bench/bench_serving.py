#!/usr/bin/env python3
"""Load generator for any OpenAI-compatible server (pagedserve, vLLM, ...).

Two scenarios, matching the two claims the project makes:

  throughput  Many independent requests with varied prompt and output lengths
              (ignore_eos makes output lengths exact). Reports output tokens/s,
              TTFT, time per output token and end-to-end latency. Run it once
              against `pagedserve` and once against `pagedserve --static-batching`.
              Lengths are either uniform random (--prompt-len/--output-len) or
              taken from real conversations in the ShareGPT dataset
              (--dataset-path + --tokenizer), the workload vLLM's benchmarks use:
              prompt = first user message, output length = token length of the
              real reply, so output lengths have a realistic long tail.

  multiturn   Concurrent chat conversations that share one long system prompt;
              every turn resends the whole history, as chat clients do.
              Reports TTFT per turn and how many prompt tokens came from the
              prefix cache. Run it with and without `--no-prefix-caching`.

Examples:
  python bench/bench_serving.py --scenario throughput --num-requests 300 --output-len 16:512
  python bench/bench_serving.py --scenario throughput --num-requests 1000 \
      --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer models/qwen2.5-0.5b-instruct
  python bench/bench_serving.py --scenario multiturn --conversations 32 --turns 6 --out mt.json
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

WORDS = ("the of and to in is was for on that with as by at from his her an were which are this be had not but "
         "they have one all their there been has more when will would who so no she other its may these about "
         "than time into only some could new them man out first then made like over did years most after also "
         "many before must through back where much your way well down should because each just those people how "
         "too little state good very make world still own see men work long get here between both life being "
         "under never day same another know while last might us great old year off come since against go came "
         "right used take three system river garden engine market signal pattern window history planet music "
         "language memory battery theory forest silver doctor island bridge camera winter summer letter").split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def parse_range(s):
    lo, hi = s.split(":")
    return int(lo), int(hi)


async def stream_request(session, url, payload):
    """Sends one streaming request. Returns timing info and usage."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    ttft, chunks, usage, text = None, 0, None, []
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = ch.get("delta", {}).get("content") if "delta" in ch else ch.get("text")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    text.append(piece)
    e2e = time.perf_counter() - t0
    out_tokens = usage["completion_tokens"] if usage else chunks
    tpot = (e2e - ttft) / (out_tokens - 1) if ttft is not None and out_tokens > 1 else None
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "usage": usage, "text": "".join(text)}


async def get_stats(session, base):
    try:
        async with session.get(base + "/stats") as r:
            return await r.json() if r.status == 200 else {}
    except aiohttp.ClientError:
        return {}


def sharegpt_requests(args, rng):
    """(prompt, output_len) pairs from ShareGPT, filtered like vLLM's benchmark."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.dataset_path) as f:
        data = json.load(f)
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    rng.shuffle(data)
    out = []
    for d in data:
        prompt, reply = d["conversations"][0]["value"], d["conversations"][1]["value"]
        p_len = len(tok(prompt).input_ids)
        o_len = len(tok(reply).input_ids)
        if p_len < 4 or o_len < 4 or p_len > 1024 or p_len + o_len > 2048:
            continue
        out.append((prompt, o_len))
        if len(out) == args.num_requests:
            break
    return out


async def run_throughput(args, session, base):
    rng = random.Random(args.seed)
    if args.dataset_path:
        pairs = sharegpt_requests(args, rng)
    else:
        plo, phi = parse_range(args.prompt_len)
        olo, ohi = parse_range(args.output_len)
        pairs = [(words(rng, rng.randint(plo, phi)), rng.randint(olo, ohi)) for _ in range(args.num_requests)]
    reqs = [{"model": args.model, "prompt": p, "max_tokens": n, "temperature": 0.0, "ignore_eos": True}
            for p, n in pairs]
    url = base + "/v1/completions"
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def one(i, payload):
        if args.request_rate != float("inf"):
            await asyncio.sleep(arrivals[i])
        if sem:
            async with sem:
                return await stream_request(session, url, payload)
        return await stream_request(session, url, payload)

    t = 0.0
    arrivals = []
    for _ in reqs:
        arrivals.append(t)
        if args.request_rate != float("inf"):
            t += rng.expovariate(args.request_rate)
    s0 = await get_stats(session, base)
    t0 = time.perf_counter()
    results = await asyncio.gather(*[one(i, r) for i, r in enumerate(reqs)])
    dur = time.perf_counter() - t0
    s1 = await get_stats(session, base)
    out_tok = sum(r["usage"]["completion_tokens"] for r in results)
    in_tok = sum(r["usage"]["prompt_tokens"] for r in results)
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if r["tpot"] is not None]
    e2es = [r["e2e"] for r in results]
    out_lens = sorted(r["max_tokens"] for r in reqs)
    summary = {
        "scenario": "throughput", "workload": "sharegpt" if args.dataset_path else "uniform",
        "output_len_p50": pct(out_lens, 50), "output_len_p95": pct(out_lens, 95), "output_len_max": out_lens[-1],
        "requests": len(results), "duration_s": dur,
        "input_tokens": in_tok, "output_tokens": out_tok,
        "output_tokens_per_s": out_tok / dur, "requests_per_s": len(results) / dur,
        "ttft_p50_ms": 1e3 * pct(ttfts, 50), "ttft_p95_ms": 1e3 * pct(ttfts, 95), "ttft_p99_ms": 1e3 * pct(ttfts, 99),
        "tpot_mean_ms": 1e3 * statistics.mean(tpots) if tpots else float("nan"),
        "e2e_p50_s": pct(e2es, 50), "e2e_p95_s": pct(e2es, 95),
    }
    if s0 and s1:
        steps = s1["steps"] - s0["steps"]
        summary["server_steps"] = steps
        summary["avg_tokens_per_step"] = ((s1["generated_tokens"] - s0["generated_tokens"]) +
                                          (s1["prefill_tokens"] - s0["prefill_tokens"])) / max(1, steps)
        summary["preemptions"] = s1["num_preemptions"] - s0["num_preemptions"]
        # Hardware-independent batching efficiency: sequences decoded per step.
        summary["avg_decode_batch"] = (s1["generated_tokens"] - s0["generated_tokens"]) / max(1, steps)
    return summary


async def run_multiturn(args, session, base):
    rng = random.Random(args.seed)
    system = "You are the assistant for a large online store. " + words(rng, args.system_words)
    ulo, uhi = parse_range(args.user_words)
    url = base + "/v1/chat/completions"
    per_turn = [[] for _ in range(args.turns)]
    cached = [[] for _ in range(args.turns)]
    prompt_tokens = [[] for _ in range(args.turns)]

    async def conversation(cid):
        crng = random.Random(args.seed * 1000 + cid)
        await asyncio.sleep(crng.uniform(0, args.stagger))
        history = [{"role": "system", "content": system}]
        for t in range(args.turns):
            history.append({"role": "user", "content": words(crng, crng.randint(ulo, uhi)) + "?"})
            r = await stream_request(session, url, {"model": args.model, "messages": history,
                                                    "max_tokens": args.max_tokens, "temperature": 0.0,
                                                    "ignore_eos": True})
            per_turn[t].append(r["ttft"])
            u = r["usage"]
            prompt_tokens[t].append(u["prompt_tokens"])
            cached[t].append((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            history.append({"role": "assistant", "content": r["text"]})
            if args.think_time:
                await asyncio.sleep(crng.expovariate(1.0 / args.think_time))

    s0 = await get_stats(session, base)
    t0 = time.perf_counter()
    await asyncio.gather(*[conversation(c) for c in range(args.conversations)])
    dur = time.perf_counter() - t0
    s1 = await get_stats(session, base)
    all_ttft = [x for turn in per_turn for x in turn if x is not None]
    later = [x for turn in per_turn[1:] for x in turn if x is not None]
    summary = {
        "scenario": "multiturn", "conversations": args.conversations, "turns": args.turns, "duration_s": dur,
        "ttft_p50_ms": 1e3 * pct(all_ttft, 50), "ttft_p95_ms": 1e3 * pct(all_ttft, 95),
        "ttft_p95_ms_turn2plus": 1e3 * pct(later, 95),
        "prompt_tokens": sum(map(sum, prompt_tokens)), "cached_prompt_tokens": sum(map(sum, cached)),
        "per_turn": [{"turn": t + 1, "ttft_p50_ms": 1e3 * pct(per_turn[t], 50),
                      "ttft_p95_ms": 1e3 * pct(per_turn[t], 95),
                      "avg_prompt_tokens": statistics.mean(prompt_tokens[t]),
                      "avg_cached_tokens": statistics.mean(cached[t])} for t in range(args.turns)],
    }
    summary["cache_hit_rate"] = summary["cached_prompt_tokens"] / max(1, summary["prompt_tokens"])
    if s0 and s1:
        summary["server_prefill_tokens"] = s1["prefill_tokens"] - s0["prefill_tokens"]
    return summary


async def main_async(args):
    base = args.base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if not args.model:
            async with session.get(base + "/v1/models") as r:
                args.model = (await r.json())["data"][0]["id"]
        if args.scenario == "throughput":
            return await run_throughput(args, session, base)
        return await run_multiturn(args, session, base)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="")
    p.add_argument("--scenario", choices=["throughput", "multiturn"], default="throughput")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="write the summary JSON here")
    g = p.add_argument_group("throughput")
    g.add_argument("--num-requests", type=int, default=200)
    g.add_argument("--prompt-len", default="32:256", help="prompt length in words, lo:hi")
    g.add_argument("--output-len", default="16:512", help="max_tokens, lo:hi (exact, ignore_eos)")
    g.add_argument("--request-rate", type=float, default=float("inf"), help="req/s (Poisson); inf = all at once")
    g.add_argument("--max-concurrency", type=int, default=0)
    g.add_argument("--dataset-path", default="", help="ShareGPT JSON (ShareGPT_V3_unfiltered_cleaned_split.json)")
    g.add_argument("--tokenizer", default="", help="HF tokenizer dir, used to measure ShareGPT lengths")
    g = p.add_argument_group("multiturn")
    g.add_argument("--conversations", type=int, default=32)
    g.add_argument("--turns", type=int, default=5)
    g.add_argument("--system-words", type=int, default=600)
    g.add_argument("--user-words", default="20:60")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--stagger", type=float, default=2.0, help="spread conversation starts over N seconds")
    g.add_argument("--think-time", type=float, default=0.0, help="mean pause between turns (s)")
    args = p.parse_args()
    if args.dataset_path and not args.tokenizer:
        p.error("--dataset-path needs --tokenizer (the model directory)")
    summary = asyncio.run(main_async(args))
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
