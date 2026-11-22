"""One-time setup: fetch the pinned Qwen2.5 tokenizer.json into models/ (D-010, D-021).

Run:  python -m rag_chat.download_tokenizer

This is the only step that touches the network for the tokenizer; at runtime
tokens.py loads the local file with Tokenizer.from_file. The revision is pinned
to a Hugging Face commit and the file is checked against a SHA-256, so every
machine counts tokens with byte-identical vocab and merges.
"""
import hashlib
import sys

import requests

from rag_chat import config

URL = (
    f"https://huggingface.co/{config.TOKENIZER_REPO}"
    f"/resolve/{config.TOKENIZER_REVISION}/tokenizer.json"
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    dest = config.TOKENIZER_PATH

    if dest.exists() and sha256_of(dest.read_bytes()) == config.TOKENIZER_SHA256:
        print(f"Tokenizer already present and verified: {dest}")
        return 0

    print(f"Downloading {URL}")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()

    digest = sha256_of(resp.content)
    if digest != config.TOKENIZER_SHA256:
        print(
            f"SHA-256 mismatch: expected {config.TOKENIZER_SHA256}, got {digest}. "
            "Not saving.",
            file=sys.stderr,
        )
        return 1

    # Write to a temp file then rename, so an interrupted run never leaves a
    # half-written tokenizer that tokens.py would try to load.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)

    print(f"Saved {len(resp.content):,} bytes to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
