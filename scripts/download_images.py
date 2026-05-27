#!/usr/bin/env python3
"""自动下载角色参考图片 - 使用 Bing 图片搜索。

遍历 data/characters/ 下的所有角色，用角色名从 Bing 搜图下载。
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from icrawler.builtin import BingImageCrawler


def download_images(
    characters_dir: str = "data/characters",
    max_per_character: int = 5,
):
    chars_path = Path(characters_dir)
    if not chars_path.exists():
        print(f"ERROR: Characters directory not found: {characters_dir}")
        return

    card_files = sorted(chars_path.glob("*/card.json"))
    if not card_files:
        print("No character cards found.")
        return

    success_total = 0

    for card_path in card_files:
        card = json.loads(card_path.read_text())
        char_id = card["id"]
        char_name = card.get("name", char_id)

        images_dir = card_path.parent / "images"
        images_dir.mkdir(exist_ok=True)

        # 删除 .gitkeep
        gitkeep = images_dir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()

        # 检查已有图片数量
        existing_imgs = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        if len(existing_imgs) >= max_per_character:
            print(f"SKIP {char_name}: already has {len(existing_imgs)} images")
            success_total += len(existing_imgs)
            continue

        needed = max_per_character - len(existing_imgs)
        print(f"Downloading {char_name} ({char_id}): need {needed} images...")

        # 用临时目录存 Bing 下载的图片（Bing 会按序号命名）
        tmp_dir = Path(f"/tmp/lookalike_dl/{char_id}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        try:
            crawler = BingImageCrawler(
                downloader_threads=2,
                storage={"root_dir": str(tmp_dir)},
            )
            crawler.crawl(
                keyword=f"{char_name} character",
                max_num=needed + 3,  # 多下几张防止有坏的
                min_size=(100, 100),
            )
        except Exception as e:
            print(f"  WARN {char_name}: crawler error: {e}")

        # 把下载的文件移到角色 images 目录
        downloaded = 0
        for tmp_file in sorted(tmp_dir.glob("*")):
            if tmp_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                filename = f"{len(existing_imgs) + downloaded + 1:02d}.jpg"
                shutil.move(str(tmp_file), str(images_dir / filename))
                downloaded += 1
                if downloaded >= needed:
                    break
            except Exception:
                continue

        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if downloaded == 0:
            print(f"  FAIL {char_name}: no images downloaded")
        else:
            print(f"  DONE {char_name}: {downloaded} downloaded, now has {len(existing_imgs) + downloaded} total")
            success_total += len(existing_imgs) + downloaded

    print(f"\nDone. Total images across all characters: {success_total}")


def main():
    parser = argparse.ArgumentParser(
        description="Download character reference images via Bing image search"
    )
    parser.add_argument("--characters-dir", default="data/characters")
    parser.add_argument("--max", type=int, default=5, help="Max images per character")
    args = parser.parse_args()

    download_images(
        characters_dir=args.characters_dir,
        max_per_character=args.max,
    )


if __name__ == "__main__":
    main()
