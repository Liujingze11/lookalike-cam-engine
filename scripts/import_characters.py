#!/usr/bin/env python3
"""批量导入角色：将含 card.json 的目录复制到 data/characters/ 下。"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Import character cards into the library")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Source directory containing character subdirectories with card.json",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/characters",
        help="Target character library directory (default: data/characters)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing characters",
    )
    args = parser.parse_args()

    source = Path(args.source_dir)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0

    for card_path in sorted(source.glob("*/card.json")):
        char_dir = card_path.parent
        char_id = card_path.parent.name
        dest_dir = target / char_id

        if dest_dir.exists() and not args.overwrite:
            print(f"SKIP {char_id}: already exists (use --overwrite)")
            skipped += 1
            continue

        # 验证 card.json 格式
        try:
            card = json.loads(card_path.read_text())
            assert "id" in card, "Missing 'id' field"
            assert "name" in card, "Missing 'name' field"
            assert card["id"] == char_id, f"id '{card['id']}' != directory name '{char_id}'"
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"ERROR {char_id}: invalid card.json - {e}")
            continue

        # 复制整个目录
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(char_dir, dest_dir)
        print(f"IMPORTED {char_id}: {card.get('name', char_id)}")
        imported += 1

    print(f"\nDone. Imported {imported}, skipped {skipped}.")


if __name__ == "__main__":
    main()
