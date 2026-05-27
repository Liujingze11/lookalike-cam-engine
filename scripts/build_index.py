#!/usr/bin/env python3
"""构建 FAISS 索引：遍历角色库，为每张参考图生成 embedding，建立索引。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def build_global_index(
    characters_dir: str = "data/characters",
    output_path: str = "indexes/global.index",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda",
):
    """为角色库中所有图片提取 OpenCLIP global embedding，构建 FAISS 索引。"""
    from engine.embedders.clip_embedder import ClipEmbedder
    from engine.retrievers.faiss_retriever import FaissRetriever

    chars_path = Path(characters_dir)
    if not chars_path.exists():
        print(f"ERROR: Characters directory not found: {characters_dir}")
        sys.exit(1)

    print(f"Loading OpenCLIP model ({model_name}, {pretrained}) on {device}...")
    embedder = ClipEmbedder(model_name=model_name, pretrained=pretrained, device=device)

    all_vectors = []
    all_ids = []
    stats = {"characters": 0, "images": 0, "skipped": 0}

    for card_path in sorted(chars_path.glob("*/card.json")):
        card = json.loads(card_path.read_text())
        char_id = card["id"]

        # 跳过不安全角色
        if not card.get("safety", {}).get("allowed", True):
            print(f"SKIP {char_id}: blocked by safety policy")
            stats["skipped"] += 1
            continue

        images_dir = card_path.parent / "images"
        if not images_dir.exists():
            print(f"WARN {char_id}: no images/ directory")
            continue

        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if not image_files:
            print(f"WARN {char_id}: no images found in images/")
            continue

        char_vectors = []
        for img_path in image_files:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                fv = embedder.embed(img)
                char_vectors.append(fv.vector)
            except Exception as e:
                print(f"WARN {char_id}/{img_path.name}: {e}")
                continue

        if not char_vectors:
            continue

        # 对同一角色的多张图取平均作为该角色的代表向量
        mean_vector = np.mean(char_vectors, axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)

        all_vectors.append(mean_vector)
        all_ids.append(char_id)
        stats["characters"] += 1
        stats["images"] += len(char_vectors)
        print(f"  {char_id}: {len(char_vectors)} images -> embedding ({embedder.dimension}d)")

    if not all_vectors:
        print("ERROR: No valid characters found. Nothing to index.")
        sys.exit(1)

    vec_matrix = np.stack(all_vectors, axis=0)
    print(f"\nBuilding FAISS index: {vec_matrix.shape[0]} vectors x {vec_matrix.shape[1]} dims")

    retriever = FaissRetriever(dimension=embedder.dimension)
    retriever.build_index(vec_matrix, all_ids)
    retriever.save(output_path)

    print(f"Index saved to {output_path}")
    print(f"Stats: {stats['characters']} characters, {stats['images']} images, {stats['skipped']} skipped")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from character library")
    parser.add_argument("--characters-dir", default="data/characters")
    parser.add_argument("--output", default="indexes/global.index")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    build_global_index(
        characters_dir=args.characters_dir,
        output_path=args.output,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )


if __name__ == "__main__":
    main()
