#!/usr/bin/env python3
"""S.5(e): raw-socket SSE framing assertions against a live qwen-serve.

Speaks HTTP/1.1 over a plain socket (no client library) and asserts the exact
wire shape: status line, Content-Type: text/event-stream, every frame is
`data: <json>\n\n`, first chunk carries delta {role:"assistant", content:""},
exactly one finish_reason chunk, usage rules per stream_options.include_usage
(usage:null on every chunk + trailing usage-only chunk with empty choices),
and the stream terminates with `data: [DONE]`.

Usage: sse_capture.py [host] [port]   (default 127.0.0.1 8000)
"""

import json
import socket
import sys

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def raw_sse(path: str, body: dict) -> tuple[str, list[str]]:
    """POST body, return (headers_text, list of data-frame payloads)."""
    payload = json.dumps(body).encode()
    req = (
        f"POST {path} HTTP/1.1\r\nHost: {HOST}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode() + payload
    with socket.create_connection((HOST, PORT), timeout=600) as s:
        s.sendall(req)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    headers = head.decode()
    # chunked transfer-encoding: strip chunk-size lines
    if "chunked" in headers.lower():
        body_bytes = b""
        while rest:
            size_line, _, rest = rest.partition(b"\r\n")
            size = int(size_line, 16)
            if size == 0:
                break
            body_bytes += rest[:size]
            rest = rest[size + 2 :]
    else:
        body_bytes = rest
    text = body_bytes.decode()
    frames = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        # keep-alive comments start with ':'
        if block.startswith(":"):
            continue
        assert block.startswith("data: "), f"non-data SSE block: {block!r}"
        frames.append(block[len("data: ") :])
    return headers, frames


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main() -> None:
    model = json.loads(
        __import__("urllib.request", fromlist=["urlopen"]).urlopen(
            f"http://{HOST}:{PORT}/v1/models"
        ).read()
    )["data"][0]["id"]
    print(f"SSE framing gate against model={model}")

    for include_usage in (False, True):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": True,
            "temperature": 0,
            "max_tokens": 16,
        }
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        headers, frames = raw_sse("/v1/chat/completions", body)
        print(f"- include_usage={include_usage}: {len(frames)} frames")

        check("status 200", headers.startswith("HTTP/1.1 200"))
        check("content-type event-stream", "text/event-stream" in headers.lower())
        check("terminates with [DONE]", frames[-1] == "[DONE]")
        chunks = [json.loads(f) for f in frames[:-1]]
        check("all chunks parse as JSON", True)
        check(
            "object is chat.completion.chunk",
            all(c["object"] == "chat.completion.chunk" for c in chunks),
        )
        first = chunks[0]
        d0 = first["choices"][0]["delta"]
        check("first chunk role=assistant", d0.get("role") == "assistant")
        check("first chunk content==''", d0.get("content") == "")
        finish = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
        check("exactly one finish chunk", len(finish) == 1, f"got {len(finish)}")
        ids = {c["id"] for c in chunks}
        check("stable chunk id", len(ids) == 1)
        if include_usage:
            with_choices = [c for c in chunks if c["choices"]]
            check(
                "usage:null on every non-final chunk",
                all("usage" in c and c["usage"] is None for c in with_choices),
            )
            last = chunks[-1]
            check("final usage chunk has empty choices", last["choices"] == [])
            check(
                "final usage chunk counts",
                last["usage"]["total_tokens"]
                == last["usage"]["prompt_tokens"] + last["usage"]["completion_tokens"],
            )
        else:
            check("no usage key anywhere", all("usage" not in c for c in chunks))
    print("SSE framing gate: ALL PASS")


if __name__ == "__main__":
    main()
