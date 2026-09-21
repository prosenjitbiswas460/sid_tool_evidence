#!/usr/bin/env python3
"""Quick sanity check for semantic ID artifacts produced by build_semantic_ids.py."""

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", required=True)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    item_id_to_sid = torch.load(os.path.join(artifact_dir, "item_id_to_sid.pt"), map_location="cpu")
    sid_map = torch.load(os.path.join(artifact_dir, "sid_map.pt"), map_location="cpu")

    with open(os.path.join(artifact_dir, "metadata.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    sample_items = list(item_id_to_sid.items())[:3]
    print("metadata:", json.dumps(metadata, indent=2))
    print("sid_map shape:", tuple(sid_map.shape))
    print("num item mappings:", len(item_id_to_sid))
    print("sample mappings:")
    for item_id, sid in sample_items:
        print(f"  {item_id} -> {sid.tolist()}")


if __name__ == "__main__":
    main()
