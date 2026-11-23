#!/usr/bin/env python3
"""
按 arXiv 官方列表补爬最近几天的论文 / Backfill recent dates from arXiv's official listings.

与 backfill.py 按投稿时间窗口推算不同，本脚本直接读取 arXiv 的公告列表，
论文集合与每日爬虫（scrapy crawl arxiv）完全一致：
- /list/{cat}/pastweek 按公告日给出最近 5 个公告日的论文（含标题/作者/备注/分类，无摘要）
- /list/{cat}/new 给出最新一天的论文及完整摘要
- 其余日期的摘要通过 arXiv API 按 id_list 批量获取

只保留主分类属于 CATEGORIES 的论文，并与前 7 天数据去重（同 check_stats.py）。
数据文件的日期即 arXiv 列表上的公告日（与每日工作流一致）。

用法 / Usage:
    CATEGORIES="cs.CV" python listing_backfill.py --dates 2026-11-19 2026-11-20 2026-11-23
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import arxiv
import requests
from scrapy import Selector

HISTORY_DAYS = 7
MAX_ATTEMPTS = 6
HEADERS = {"User-Agent": "daily-arXiv-ai-enhanced (listing backfill)"}


def get_html(url):
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                return resp.text
            print(f"  {url} -> HTTP {resp.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {url} -> {e}", file=sys.stderr)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def clean(texts):
    return " ".join(" ".join(texts).split())


def parse_entry(dt):
    link = dt.css("a[title='Abstract']::attr(href)").get()
    if not link:
        return None
    arxiv_id = link.split("/")[-1]
    dd = dt.xpath("following-sibling::dd[1]")
    primary = clean(dd.css(".list-subjects .primary-subject::text").getall())
    subjects = clean(dd.css(".list-subjects ::text").getall())
    code = r"\(([a-z\-]+(?:\.[A-Za-z\-]+)?)\)"
    comment = clean(dd.css(".list-comments ::text").getall()).replace("Comments:", "", 1).strip()
    return {
        "id": arxiv_id,
        "primary": (re.findall(code, primary) or [None])[0],
        "categories": re.findall(code, subjects),
        "title": clean(dd.css(".list-title::text").getall()),
        "authors": [clean([a]) for a in dd.css(".list-authors a::text").getall()],
        "comment": comment or None,
        "summary": clean(dd.css("p.mathjax ::text").getall()),
    }


def parse_pastweek(html):
    """返回 {公告日: [entry, ...]}"""
    by_day, current = {}, None
    for node in Selector(text=html).css("dl#articles > *"):
        tag = node.root.tag
        if tag == "h3":
            label = clean(node.css("::text").getall()).split("(")[0].strip()
            current = datetime.strptime(label, "%a, %d %b %Y").date()
            by_day.setdefault(current, [])
        elif tag == "dt" and current is not None:
            entry = parse_entry(node)
            if entry:
                by_day[current].append(entry)
    return by_day


def parse_new(html):
    """返回 (公告日, [entry, ...])，只含新提交与交叉列表，不含替换版本"""
    sel = Selector(text=html)
    header = clean(sel.css("h3::text").getall()[:1])
    day = datetime.strptime(header.replace("Showing new listings for ", ""), "%A, %d %B %Y").date()
    anchors = [int(h.split("item")[-1]) for h in sel.css("div#dlpage ul li a::attr(href)").getall() if "item" in h]
    entries = []
    for dt in sel.css("dl dt"):
        anchor = dt.css("a[name^='item']::attr(name)").get()
        if anchor and anchors and int(anchor.split("item")[-1]) >= anchors[-1]:
            continue
        entry = parse_entry(dt)
        if entry:
            entries.append(entry)
    return day, entries


def fetch_abstracts(ids):
    abstracts = {}
    for start in range(0, len(ids), 100):
        chunk = ids[start:start + 100]
        for attempt in range(MAX_ATTEMPTS):
            try:
                client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=2)
                for paper in client.results(arxiv.Search(id_list=chunk, max_results=len(chunk))):
                    abstracts[paper.get_short_id().rsplit("v", 1)[0]] = paper.summary
                break
            except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError) as e:
                wait = 30 * 2 ** attempt
                print(f"  arXiv API failed ({e}), retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            raise RuntimeError("arXiv API keeps failing, try again later")
        time.sleep(5)
    return abstracts


def load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line).get("id", "") for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD announcement dates")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    categories = [c.strip() for c in os.environ.get("CATEGORIES", "cs.CV").split(",") if c.strip()]
    targets = sorted(date.fromisoformat(d) for d in args.dates)

    listings = {}
    for cat in categories:
        for day, entries in parse_pastweek(get_html(f"https://arxiv.org/list/{cat}/pastweek?skip=0&show=2000")).items():
            listings.setdefault(day, []).extend(entries)
        new_day, new_entries = parse_new(get_html(f"https://arxiv.org/list/{cat}/new"))
        # /new 带摘要，覆盖 pastweek 中同一天的条目
        listings[new_day] = [e for e in listings.get(new_day, []) if e["id"] not in {n["id"] for n in new_entries}]
        listings[new_day].extend(new_entries)

    for day in targets:
        out = os.path.join(args.data_dir, f"{day.isoformat()}.jsonl")
        if day not in listings:
            print(f"{day}: not in arXiv's recent listings (weekend, holiday or older than 5 announcement days), skip", file=sys.stderr)
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"{day}: {out} exists, skip", file=sys.stderr)
            continue

        seen, papers = set(), []
        for entry in listings[day]:
            if entry["primary"] in categories and entry["id"] not in seen:
                seen.add(entry["id"])
                papers.append(entry)

        missing = [p["id"] for p in papers if not p["summary"]]
        if missing:
            abstracts = fetch_abstracts(missing)
            for p in papers:
                p["summary"] = p["summary"] or abstracts.get(p["id"], "")

        history_ids = set()
        for i in range(1, HISTORY_DAYS + 1):
            past = (day - timedelta(days=i)).isoformat()
            history_ids |= load_ids(os.path.join(args.data_dir, f"{past}.jsonl"))
        new_papers = [p for p in papers if p["id"] not in history_ids and p["summary"]]

        print(
            f"{day}: listed {len(listings[day])}, primary in {categories} {len(papers)}, "
            f"after dedup {len(new_papers)}",
            file=sys.stderr,
        )
        if new_papers:
            with open(out, "w", encoding="utf-8") as f:
                for p in new_papers:
                    f.write(json.dumps({
                        "id": p["id"],
                        "categories": p["categories"],
                        "pdf": f"https://arxiv.org/pdf/{p['id']}",
                        "abs": f"https://arxiv.org/abs/{p['id']}",
                        "authors": p["authors"],
                        "title": p["title"],
                        "comment": p["comment"],
                        "summary": p["summary"],
                    }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
