#!/usr/bin/env python3
"""自动下载角色参考图片 - Bing 图片搜索。

遍历 data/characters/ 下的所有角色，用多种搜索词搜图并下载。
过滤掉合照、群像，尽量只保留单人/单角色图片。
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler

# 这些关键词出现在 URL 里说明大概率是合照/全家福/拼图
GROUP_KEYWORDS = [
    "all-", "all_", "every-", "every_",
    "characters", "character-list", "character_list",
    "cast", "team", "group", "family", "bros",
    "squad", "lineup", "roster", "collection",
    "vs-", "and-", "-and-",
    "collage", "mosaic", "tier-list", "tierlist",
    "wallpaper", "banner", "poster",
    "full-cast", "full_cast",
]


def _is_group_url(url: str) -> bool:
    """根据 URL 关键词判断图片是否是群像。"""
    url_lower = url.lower()
    for kw in GROUP_KEYWORDS:
        if kw in url_lower:
            return True
    return False


def download_images(
    characters_dir: str = "data/characters",
    max_per_character: int = 5,
    force: bool = False,
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
        char_type = card.get("type", "")

        images_dir = card_path.parent / "images"
        images_dir.mkdir(exist_ok=True)

        # 删除 .gitkeep
        gitkeep = images_dir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()

        # 已有的图片
        existing_imgs = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]

        if force:
            # 清掉旧的重新下
            for f in existing_imgs:
                f.unlink()
            existing_imgs = []

        if len(existing_imgs) >= max_per_character:
            print(f"SKIP {char_name}: already has {len(existing_imgs)} images")
            success_total += len(existing_imgs)
            continue

        needed = max_per_character - len(existing_imgs)
        print(f"\n{'='*50}")
        print(f"{char_name} ({char_id}) - need {needed} images")
        print(f"{'='*50}")

        # 多个搜索词，逐个搜，直到凑够
        queries = _build_queries(char_name, char_type)
        tmp_dir = Path(f"/tmp/lookalike_dl/{char_id}")

        for qi, query in enumerate(queries):
            if needed <= 0:
                break

            print(f"  Search [{qi+1}/{len(queries)}]: '{query}'")

            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

            try:
                crawler = BingImageCrawler(
                    downloader_threads=2,
                    storage={"root_dir": str(tmp_dir)},
                )
                # 每轮多下点，过滤后会少
                crawler.crawl(
                    keyword=query,
                    max_num=min(needed * 3 + 2, 20),
                    min_size=(200, 200),
                    file_idx_offset=0,
                )
            except Exception as e:
                print(f"    WARN: crawler error: {e}")
                continue

            # 过滤并移动图片
            time.sleep(0.5)
            tmp_files = [
                f for f in sorted(tmp_dir.glob("*"))
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

            if not tmp_files:
                print(f"    no results")
                continue

            # 从已有图片数量往后编号，避免覆盖
            current_count = len([
                f for f in images_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ])
            kept = 0
            for tf in tmp_files:
                if needed <= 0:
                    break
                try:
                    filename = f"{current_count + kept + 1:02d}.jpg"
                    dest = images_dir / filename
                    shutil.move(str(tf), str(dest))
                    kept += 1
                    needed -= 1
                except Exception:
                    continue

            # 清理
            shutil.rmtree(tmp_dir, ignore_errors=True)

            if kept > 0:
                print(f"    -> kept {kept} images")
            time.sleep(0.3)

        # 最终统计
        final_imgs = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        total = len(final_imgs)
        if total == 0:
            print(f"  FAIL: no images for {char_name}")
        else:
            print(f"  DONE: {char_name} has {total} images")
            success_total += total

    print(f"\n{'='*50}")
    print(f"Total images across all characters: {success_total}")
    print(f"Next: python scripts/build_index.py --device cuda")


def _build_queries(name: str, char_type: str) -> list[str]:
    """为角色构建专门的搜索词列表，优先搜单人/官方图。"""
    queries = []

    # 按优先级排列搜索词，越精确的越靠前
    if char_type == "game_character":
        queries = [
            f'"{name}" official render png',
            f'"{name}" character art solo',
            f'"{name}" render transparent',
        ]
    elif char_type == "anime_character":
        queries = [
            f'"{name}" official art solo',
            f'"{name}" anime render png',
            f'"{name}" character design sheet',
        ]
    else:
        queries = [
            f'"{name}" solo render',
            f'"{name}" official art standalone',
        ]

    # 通用补充搜索词
    queries.extend([
        f"{name} character transparent",
        f"{name} render solo",
    ])

    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Download character reference images via Bing image search"
    )
    parser.add_argument("--characters-dir", default="data/characters")
    parser.add_argument("--max", type=int, default=5, help="Max images per character")
    parser.add_argument("--force", action="store_true", help="Re-download all images")
    args = parser.parse_args()

    download_images(
        characters_dir=args.characters_dir,
        max_per_character=args.max,
        force=args.force,
    )


if __name__ == "__main__":
    main()
