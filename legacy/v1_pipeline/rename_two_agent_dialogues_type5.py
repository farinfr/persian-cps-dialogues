#!/usr/bin/env python3
"""
Rename two-agent highest-metric dialogue JSON files to type5_dialogue_{n}_{productId}.json
"""

import os
import re

BASE_DIR = "llm_generated_dialogues_two_agent_highest_metric1"


def main() -> None:
    print("Renaming two-agent dialogue JSON files to type5_dialogue_{n}_{productId}.json")

    files = sorted(
        f for f in os.listdir(BASE_DIR)
        if f.endswith(".json") and f.startswith("dialogue_")
    )

    for fname in files:
        match = re.match(r"dialogue_(\d+)_(\d+)\.json", fname)
        if not match:
            print(f"  Skipping unexpected file name: {fname}")
            continue

        number = match.group(1)
        product_id = match.group(2)
        new_name = f"type5_dialogue_{number}_{product_id}.json"

        src = os.path.join(BASE_DIR, fname)
        dst = os.path.join(BASE_DIR, new_name)

        print(f"  {fname} -> {new_name}")
        os.rename(src, dst)

    print("✅ Renaming completed.")


if __name__ == "__main__":
    main()


