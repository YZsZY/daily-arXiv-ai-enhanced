import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()


NO_INTEREST_TEXT = "(No interest filter is configured: keep every paper.)"
DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed"
}


class PaperAnalysis(BaseModel):
    """一次调用同时完成兴趣过滤与摘要分析，字段顺序保证模型先判断再总结"""
    keep: bool = Field(description="Whether this paper matches the user's research interests and should be kept")
    reason: str = Field(description="Short reason for the keep/reject decision")
    matched_interests: List[str] = Field(
        default_factory=list,
        description="The interest keywords/themes that match this paper"
    )
    tldr: str = Field(default="", description=Structure.model_fields["tldr"].description + "; empty if keep is false")
    motivation: str = Field(default="", description=Structure.model_fields["motivation"].description + "; empty if keep is false")
    method: str = Field(default="", description=Structure.model_fields["method"].description + "; empty if keep is false")
    result: str = Field(default="", description=Structure.model_fields["result"].description + "; empty if keep is false")
    conclusion: str = Field(default="", description=Structure.model_fields["conclusion"].description + "; empty if keep is false")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def parse_interests(raw_interest: str) -> List[str]:
    """Parse INTEREST env into a clean keyword list."""
    if not raw_interest:
        return []

    raw_interest = raw_interest.strip()
    if raw_interest in {'""', "''"}:
        return []

    if (
        len(raw_interest) >= 2 and
        raw_interest[0] == raw_interest[-1] and
        raw_interest[0] in {'"', "'"}
    ):
        raw_interest = raw_interest[1:-1].strip()

    if not raw_interest:
        return []

    normalized = raw_interest.replace("\n", ";").replace(",", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def is_sensitive(content: str) -> bool:
    """
    调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
    返回 True 表示触发敏感词，False 表示未触发。
    设置 SKIP_SENSITIVE_CHECK=1 可跳过检测（本地网络无法访问该接口时使用）。
    """
    if os.environ.get("SKIP_SENSITIVE_CHECK") == "1":
        return False
    try:
        resp = requests.post(
            "https://spam.dw-dengwei.workers.dev",
            json={"text": content},
            timeout=5
        )
        if resp.status_code == 200:
            result = resp.json()
            # 约定接口返回 {"sensitive": true/false, ...}
            return result.get("sensitive", True)
        else:
            # 接口异常时按触发敏感词处理
            print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
            return True
    except Exception as e:
        print(f"Sensitive check error: {e}", file=sys.stderr)
        return True


def parse_partial_output(error_msg: str, item_id: str) -> Dict:
    """从 OutputParserException 的错误信息中尽量恢复模型输出的 JSON"""
    if "arguments:" not in error_msg:
        return {}
    try:
        json_str = error_msg.split("arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
        # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
        json_str = json_str.replace('\\', '\\\\')
        return json.loads(json_str)
    except Exception as json_e:
        print(f"Failed to parse JSON for {item_id}: {json_e}", file=sys.stderr)
        return {}


def process_single_item(chain, item: Dict, language: str, interests: List[str]) -> Optional[Dict]:
    """一次模型调用完成单篇论文的兴趣过滤与总结，未保留或触发敏感词时返回 None"""
    item_id = item.get('id', 'unknown')

    if is_sensitive(item.get("summary", "")):
        return None

    try:
        response: PaperAnalysis = chain.invoke({
            "language": language,
            "interests": "; ".join(interests) if interests else NO_INTEREST_TEXT,
            "title": item.get("title", ""),
            "categories": ", ".join(item.get("categories", [])),
            "content": item['summary'],
        })
        output = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        output = parse_partial_output(str(e), item_id)
        print(f"Using partial AI data for {item_id}: {list(output.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error for {item_id}: {e}. Keeping the paper.", file=sys.stderr)
        output = {}

    # 调用失败时 output 中没有 keep，保留论文以免漏掉相关内容
    if interests and not output.get("keep", True):
        return None

    if interests and "keep" in output:
        item["interest_filter"] = {
            "matched_interests": output.get("matched_interests", []),
            "reason": output.get("reason", ""),
        }
    item['AI'] = {
        field: output.get(field) or default
        for field, default in DEFAULT_AI_FIELDS.items()
    }

    # 检查 AI 生成的所有字段
    for v in item["AI"].values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, interests: List[str], max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name).with_structured_output(PaperAnalysis, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    if not interests:
        print("INTEREST is empty, skipping interest-based filtering", file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language, interests): idx
            for idx, item in enumerate(data)
        }

        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Filtering & summarizing"
        ):
            idx = future_to_idx[future]
            try:
                processed_data[idx] = future.result()
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {field: "Processing failed" for field in DEFAULT_AI_FIELDS}

    kept = sum(item is not None for item in processed_data)
    print(f"Kept {kept} / {len(data)} papers", file=sys.stderr)
    return processed_data

def main():
    # python enhance.py --data ../data/${today}.jsonl
    args = parse_args()
    model_name = 'deepseek-chat' # os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = 'Chinese' # os.environ.get("LANGUAGE", 'Chinese')
    interests = parse_interests(os.environ.get("INTEREST", ""))

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    processed_data = process_all_items(
        data,
        model_name,
        language,
        interests,
        args.max_workers
    )

    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

    if not any(item is not None for item in processed_data):
        print("No papers matched INTEREST after model filtering", file=sys.stderr)

if __name__ == "__main__":
    main()
