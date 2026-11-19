"""HTTP/SSE tests using the official `openai` Python client.

Starts the server on a tiny model and checks: chat + completions, streaming
vs non-streaming equality, n > 1, stop strings, context-length errors,
prefix-cache accounting in `usage`, concurrent streams, and that a client
disconnect aborts generation and frees KV blocks.

Usage: python tests/test_server.py <build_dir> <model_dir>
"""
import asyncio
import http.client
import json
import os
import socket
import subprocess
import sys
import time

from openai import AsyncOpenAI, BadRequestError, OpenAI

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail and not cond else ""))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def stats(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/stats")
    return json.loads(c.getresponse().read())


def main():
    build_dir, model_dir = sys.argv[1], sys.argv[2]
    port = free_port()
    log = open(os.path.join(build_dir, "test_server.log"), "w")
    proc = subprocess.Popen([os.path.join(build_dir, "pagedserve"), "--model", model_dir, "--port", str(port),
                             "--host", "127.0.0.1", "--block-size", "4", "--num-kv-blocks", "3000",
                             "--max-model-len", "8192", "--served-model-name", "tiny", "--dtype", "float32"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        run(port)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


def run(port):
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    greedy = dict(temperature=0, extra_body={"ignore_eos": True})
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short story about a robot learning to paint."}]

    models = client.models.list()
    check("GET /v1/models", models.data[0].id == "tiny")

    r = client.chat.completions.create(model="tiny", messages=messages, max_tokens=24, **greedy)
    full = r.choices[0].message.content
    check("chat completion", r.object == "chat.completion" and r.usage.completion_tokens == 24
          and r.choices[0].finish_reason == "length", r)

    chunks = list(client.chat.completions.create(model="tiny", messages=messages, max_tokens=24, stream=True,
                                                 stream_options={"include_usage": True}, **greedy))
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    check("chat streaming == non-streaming", streamed == full, f"{streamed!r} vs {full!r}")
    check("stream starts with role", chunks[0].choices[0].delta.role == "assistant")
    check("stream finish_reason", any(c.choices and c.choices[0].finish_reason == "length" for c in chunks))
    check("stream usage chunk", chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 24)

    c1 = client.completions.create(model="tiny", prompt="The meaning of life is", max_tokens=16, **greedy)
    cs = list(client.completions.create(model="tiny", prompt="The meaning of life is", max_tokens=16, stream=True,
                                        **greedy))
    check("completions streaming == non-streaming", "".join(c.choices[0].text for c in cs) == c1.choices[0].text)

    rn = client.chat.completions.create(model="tiny", messages=messages, max_tokens=10, n=3, **greedy)
    texts = [c.message.content for c in rn.choices]
    check("n=3 greedy choices identical", len(texts) == 3 and len(set(texts)) == 1, texts)
    check("n=3 usage counts all choices", rn.usage.completion_tokens == 30, rn.usage)
    rs = client.chat.completions.create(model="tiny", messages=messages, max_tokens=10, n=2, temperature=1.0,
                                        extra_body={"ignore_eos": True, "top_k": 50, "top_p": 0.9})
    check("n=2 sampled", len(rs.choices) == 2)

    long = client.chat.completions.create(model="tiny", messages=messages, max_tokens=60, **greedy)
    text = long.choices[0].message.content
    mid = len(text) // 2
    stop = text[mid:mid + 3]
    expected = text[:text.find(stop)]
    rstop = client.chat.completions.create(model="tiny", messages=messages, max_tokens=60, stop=[stop], **greedy)
    check("stop string (non-stream)", rstop.choices[0].message.content == expected
          and rstop.choices[0].finish_reason == "stop", f"stop={stop!r}")
    sstop = list(client.chat.completions.create(model="tiny", messages=messages, max_tokens=60, stop=[stop],
                                                stream=True, **greedy))
    check("stop string (stream, held-back text)",
          "".join(c.choices[0].delta.content or "" for c in sstop if c.choices) == expected)

    try:
        client.chat.completions.create(model="tiny", messages=messages, max_tokens=100000)
        check("context length error", False)
    except BadRequestError as e:
        check("context length error", "maximum context length" in str(e))

    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", "/v1/chat/completions", body="{not json", headers={"Content-Type": "application/json"})
    check("invalid JSON -> 400", c.getresponse().status == 400)

    # Prefix caching: a long shared system prompt is served from cache the 2nd time.
    sys_prompt = "You are a meticulous assistant. " * 40
    m2 = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "First question?"}]
    m3 = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "A different question!"}]
    a = client.chat.completions.create(model="tiny", messages=m2, max_tokens=4, **greedy)
    b = client.chat.completions.create(model="tiny", messages=m3, max_tokens=4, **greedy)
    cached = b.usage.prompt_tokens_details.cached_tokens
    check("prefix cache reports cached_tokens", cached >= 0.8 * a.usage.prompt_tokens,
          f"cached {cached} of {b.usage.prompt_tokens}")

    # Concurrency: many simultaneous streams, each must equal its sequential result.
    prompts = [f"Story number {i}: once upon a time" for i in range(6)]
    seq_results = [client.completions.create(model="tiny", prompt=p, max_tokens=20, **greedy).choices[0].text
                   for p in prompts]

    async def one(ac, p):
        out = []
        async for ch in await ac.completions.create(model="tiny", prompt=p, max_tokens=20, stream=True, **greedy):
            out.append(ch.choices[0].text)
        return "".join(out)

    async def many():
        ac = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
        jobs = [one(ac, prompts[i % len(prompts)]) for i in range(48)]
        return await asyncio.gather(*jobs)

    t0 = time.time()
    res = asyncio.run(many())
    ok = all(r == seq_results[i % len(prompts)] for i, r in enumerate(res))
    check(f"48 concurrent streams match sequential ({time.time() - t0:.1f}s)", ok)

    # Disconnect mid-stream: generation must stop and blocks must be released.
    before = stats(port)
    body = json.dumps({"model": "tiny", "prompt": "Tell me everything.", "max_tokens": 4000, "stream": True,
                       "ignore_eos": True, "temperature": 0})
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(f"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(body)}\r\n\r\n{body}".encode())
    head = s.recv(4096)
    time.sleep(0.3)
    s.close()
    freed = False
    for _ in range(50):
        st = stats(port)
        if st["num_running"] == 0 and st["kv_blocks_in_use"] == 0:
            freed = True
            break
        time.sleep(0.1)
    gen = st["generated_tokens"] - before["generated_tokens"]
    check("client disconnect aborts and frees KV blocks",
          head.startswith(b"HTTP/1.1 200") and freed and 0 < gen < 4000, f"generated {gen}, stats {st}")


if __name__ == "__main__":
    main()
