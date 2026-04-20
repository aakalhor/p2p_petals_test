#!/usr/bin/env python3
"""
Model presets and block-range helpers for private Petals swarms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    key: str
    model_name: str
    total_blocks: int
    summary: str


MODEL_PRESETS = {
    "bloom-7b1": ModelPreset(
        key="bloom-7b1",
        model_name="bigscience/bloom-7b1",
        total_blocks=30,
        summary="BLOOM-7.1B, practical private-swarm demo target",
    ),
    "llama-7b-legacy": ModelPreset(
        key="llama-7b-legacy",
        model_name="huggyllama/llama-7b",
        total_blocks=32,
        summary="Legacy llama-7b path kept for historical comparison",
    ),
}

DEFAULT_MODEL_PRESET = "bloom-7b1"


def resolve_model_config(preset_key: str, model_name: str | None = None, total_blocks: int | None = None) -> dict:
    preset = MODEL_PRESETS[preset_key]
    return {
        "preset": preset.key,
        "model_name": model_name or preset.model_name,
        "total_blocks": total_blocks or preset.total_blocks,
        "summary": preset.summary,
    }


def partition_blocks(total_blocks: int, num_servers: int) -> list[tuple[int, int]]:
    if num_servers <= 0:
        raise ValueError("num_servers must be positive")

    base_size = total_blocks // num_servers
    remainder = total_blocks % num_servers
    block_ranges = []
    start = 0

    for index in range(num_servers):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        block_ranges.append((start, end))
        start = end

    return block_ranges


def build_profile(profile: str, total_blocks: int) -> dict:
    servers_by_profile = {
        "single": 1,
        "compact": 2,
        "presentation": 4,
    }
    num_servers = servers_by_profile[profile]
    block_ranges = partition_blocks(total_blocks, num_servers)
    sizes = ", ".join(str(end - start) for start, end in block_ranges)
    return {
        "name": profile,
        "num_servers": num_servers,
        "block_ranges": block_ranges,
        "description": f"{num_servers} peer(s), block split [{sizes}]",
    }


def safe_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")
