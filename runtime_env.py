"""Helpers for WSL-friendly runtime defaults."""

import os
from pathlib import Path


def configure_runtime_env():
    """Keep Hugging Face cache and temp files on the Linux filesystem when possible."""
    home = Path.home()
    hf_home = home / ".cache" / "huggingface"
    hub_cache = hf_home / "hub"

    os.environ.setdefault("TMPDIR", "/tmp")
    os.environ.setdefault("TMP", os.environ["TMPDIR"])
    os.environ.setdefault("TEMP", os.environ["TMPDIR"])
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
