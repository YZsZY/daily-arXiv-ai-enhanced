#!/usr/bin/env python3
"""
补爬历史日期的 arXiv 论文 / Backfill arXiv papers for past dates.

`scrapy crawl arxiv` 只能抓取 /list/{cat}/new（当天公告），无法回溯。
本脚本用 arXiv API 按投稿时间窗口查询，复现每个 UTC 日期的 /new 列表：
arXiv 在美东时间周日至周四 20:00 公告，收录截至当天 14:00 (ET) 的投稿，
因此 UTC 日期 D（周一至周五）对应的投稿窗口为
[前前一个工作日 14:00 ET, 前一个工作日 14:00 ET)。

用法 / Usage:
    CATEGORIES="cs.CV, cs.GR" python backfill.py --start 2026-10-13 --end 2026-11-03
"""
import argparse
import json
import os
import sys
import time as time_mod
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import arxiv

ET = ZoneInfo("America/New_York")
CUTOFF = time(14, 0)
HISTORY_DAYS = 7
MAX_ATTEMPTS = 6


def prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def submission_window(file_date: date):
    """返回 UTC 日期 file_date 对应的投稿时间窗口 (UTC)，周末返回 None"""
    if file_date.weekday() >= 5:
        return None
    end_day = prev_weekday(file_date)
    start_day = prev_weekday(end_day)
    start = datetime.combine(start_day, CUTOFF, ET).astimezone(timezone.utc)
    end = datetime.combine(end_day, CUTOFF, ET).astimezone(timezone.utc)
    return start, end


def fetch_papers(categories, start: datetime, end: datetime):
    cat_query = " OR ".join(f"cat:{c}" for c in sorted(categories))
    # API 的 submittedDate 区间两端都包含，精度到分钟
    end_inclusive = end - timedelta(minutes=1)
    query = (
        f"({cat_query}) AND submittedDate:"
        f"[{start:%Y%m%d%H%M} TO {end_inclusive:%Y%m%d%H%M}]"
    )
    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    # arXiv API 会间歇性返回 429，按整次查询做指数退避重试
    for attempt in range(MAX_ATTEMPTS):
        try:
            results = list(arxiv.Client(page_size=1000, delay_seconds=5, num_retries=2).results(search))
            break
        except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError) as e:
            wait = 30 * 2 ** attempt
            print(f"  请求失败 / request failed ({e}), retry in {wait}s", file=sys.stderr)
            time_mod.sleep(wait)
    else:
        raise RuntimeError(f"arXiv API failed after {MAX_ATTEMPTS} attempts: {query}")

    papers = []
    for paper in results:
        # 与爬虫一致：只保留主分类属于目标分类的论文
        if paper.primary_category not in categories:
            continue
        arxiv_id = paper.get_short_id().rsplit("v", 1)[0]
        papers.append({
            "id": arxiv_id,
            "categories": paper.categories,
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
            "abs": f"https://arxiv.org/abs/{arxiv_id}",
            "authors": [a.name for a in paper.authors],
            "title": paper.title,
            "comment": paper.comment,
            "summary": paper.summary,
        })
    return query, papers


def load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line).get("id", "") for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), inclusive")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    categories = {c.strip() for c in os.environ.get("CATEGORIES", "cs.CV").split(",") if c.strip()}
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    d = start_date
    while d <= end_date:
        out = os.path.join(args.data_dir, f"{d.isoformat()}.jsonl")
        window = submission_window(d)
        if window is None:
            print(f"{d} 周末无公告，跳过 / weekend, skip", file=sys.stderr)
        elif os.path.exists(out) and not args.overwrite:
            print(f"{d} 已存在，跳过 / exists, skip: {out}", file=sys.stderr)
        else:
            query, papers = fetch_papers(categories, *window)
            time_mod.sleep(5)
            seen, unique = set(), []
            for p in papers:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    unique.append(p)

            history_ids = set()
            for i in range(1, HISTORY_DAYS + 1):
                past = (d - timedelta(days=i)).isoformat()
                history_ids |= load_ids(os.path.join(args.data_dir, f"{past}.jsonl"))
            new_papers = [p for p in unique if p["id"] not in history_ids]

            print(
                f"{d} window {window[0]:%m-%d %H:%M}~{window[1]:%m-%d %H:%M} UTC | "
                f"fetched {len(papers)}, unique {len(unique)}, after dedup {len(new_papers)}",
                file=sys.stderr,
            )
            if new_papers:
                with open(out, "w", encoding="utf-8") as f:
                    for p in new_papers:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
