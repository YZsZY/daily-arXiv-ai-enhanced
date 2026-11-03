#!/usr/bin/env python3
"""
为论文抓取 teaser 图并生成缩略图 / Fetch teaser figures and build thumbnails.

从 arXiv HTML 版论文中取第一张插图（通常是 Figure 1 / teaser），
压缩为 WebP 缩略图保存到 assets/teasers/{id}.webp，并写回 jsonl：
    teaser:     缩略图相对站点根目录的路径
    teaser_src: arXiv 上的原图地址（详情页展示大图用）

用法 / Usage:
    python teaser.py --data ../data/2026-11-03_AI_enhanced_Chinese.jsonl
"""
import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
from PIL import Image, ImageFile

# 论文插图分辨率常超过 Pillow 默认的解压炸弹阈值
Image.MAX_IMAGE_PIXELS = None
# arXiv 上部分原图本身就不完整，按浏览器的方式解码已有部分
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEASER_DIR = "assets/teasers"
THUMB_SIZE = (480, 720)
MAX_IMAGE_BYTES = 40 * 1024 * 1024
HEADERS = {"User-Agent": "daily-arXiv-ai-enhanced (teaser thumbnails)"}

FIGURE_RE = re.compile(r'<figure[^>]*class="ltx_figure[^"]*"(.*?)</figure>', re.S)
IMG_RE = re.compile(r'<img[^>]*\ssrc="([^"]+)"')


def get(url, **kwargs):
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
        except requests.RequestException:
            resp = None
        if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
            return resp
        time.sleep(5 * (attempt + 1))
    return resp


def find_teaser_url(arxiv_id):
    resp = get(f"https://arxiv.org/html/{arxiv_id}")
    if resp is None or resp.status_code != 200:
        return None
    for figure in FIGURE_RE.finditer(resp.text):
        for match in IMG_RE.finditer(figure.group(1)):
            src = match.group(1)
            # 跳过内联小图标和 Pillow 无法解码的 SVG
            if src.startswith("data:") or src.lower().endswith(".svg"):
                continue
            return urljoin(resp.url, src)
    return None


def make_thumbnail(content, out_path):
    img = Image.open(io.BytesIO(content))
    img.draft("RGB", (THUMB_SIZE[0] * 2, THUMB_SIZE[1] * 2))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.getchannel("A"))
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    img.save(out_path, "WEBP", quality=80, method=6)


def process(item):
    arxiv_id = item["id"]
    rel_path = f"{TEASER_DIR}/{arxiv_id}.webp"
    out_path = os.path.join(ROOT, rel_path)
    if os.path.exists(out_path) and item.get("teaser") == rel_path:
        return "cached"
    try:
        src = find_teaser_url(arxiv_id)
        if not src:
            return "no_figure"
        resp = get(src)
        if resp is None or resp.status_code != 200 or len(resp.content) > MAX_IMAGE_BYTES:
            return "download_failed"
        make_thumbnail(resp.content, out_path)
    except Exception as e:
        print(f"Teaser failed for {arxiv_id}: {e}", file=sys.stderr)
        return "error"
    item["teaser"] = rel_path
    item["teaser_src"] = src
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="AI enhanced jsonl file")
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, TEASER_DIR), exist_ok=True)
    with open(args.data, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        statuses = list(executor.map(process, items))

    tmp_path = args.data + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    os.replace(tmp_path, args.data)

    summary = {s: statuses.count(s) for s in sorted(set(statuses))}
    print(f"Teasers for {os.path.basename(args.data)}: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
