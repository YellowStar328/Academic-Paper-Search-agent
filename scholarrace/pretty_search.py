"""Pretty-print search results for human reading."""

import json
import sys
import urllib.request

QUERY = sys.argv[1] if len(sys.argv) > 1 else "transformer attention mechanism survey"
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
URL = "http://127.0.0.1:8000/api/search"

payload = json.dumps({
    "query": QUERY,
    "max_results": TOP_K,
}).encode("utf-8")

req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
except urllib.error.URLError as e:
    print(f"无法连接服务: {e}")
    print("请先启动: .venv/bin/uvicorn app.main:app --port 8000")
    sys.exit(1)
except Exception as e:
    print(f"请求失败: {e}")
    sys.exit(1)

print("=" * 90)
print(f"  搜索查询: {data['query']}")
print(f"  论文总数: {data['total_papers']}")
print(f"  耗时: {data['latency_ms']:.0f} ms ({data['latency_ms']/1000:.1f} 秒)")

# Summary
if data.get("summary"):
    s = data["summary"]
    print(f"  意图: {s.get('intent', '')}  |  领域: {s.get('domain', '')}")
    print(f"  主题聚类: {s.get('clusters_count', 0)} 个")

# Metrics — token usage, models, sources
if data.get("metrics"):
    m = data["metrics"]
    print(f"  ── 调用统计 ──")
    print(f"  LLM 调用次数: {m.get('llm_calls', 0)}")
    print(f"  Token 总用量: {m.get('token_usage', 0)}")
    print(f"  使用的模型: {', '.join(m.get('models_used', []))}")
    print(f"  数据源: {', '.join(m.get('search_sources_used', []))}")
    print(f"  论文采集: {m.get('papers_collected', 0)} → "
          f"去重后: {m.get('papers_after_dedup', 0)} → "
          f"重排后: {m.get('papers_after_rerank', 0)} → "
          f"最终: {m.get('papers_final', 0)}")
    allocs = m.get("thompson_allocations", {})
    if allocs:
        alloc_str = " | ".join(f"{k}: {v}" for k, v in allocs.items())
        print(f"  Thompson 分配: {alloc_str}")
    stages = m.get("stage_latencies", {})
    if stages:
        print(f"  ── 各阶段耗时 ──")
        for stage, ms in sorted(stages.items(), key=lambda x: x[1], reverse=True):
            print(f"    {stage}: {ms:.0f} ms")

print("=" * 90)

for i, p in enumerate(data["papers"], 1):
    print(f"\n{'━' * 90}")
    print(f"  [{i}] {p['title']}")
    print(f"{'━' * 90}")
    print(f"  年份: {p['year']}  |  引用数: {p['citation_count']}  |  来源: {p['source']}")
    authors = ", ".join(p.get("authors", [])[:5])
    if len(p.get("authors", [])) > 5:
        authors += f" 等 {len(p['authors'])} 人"
    print(f"  作者: {authors}")
    if p.get("arxiv_id"):
        print(f"  arXiv: {p['arxiv_id']}")
    if p.get("doi"):
        print(f"  DOI:   {p['doi']}")
    if p.get("url"):
        print(f"  链接:  {p['url']}")
    if p.get("pdf_url"):
        print(f"  PDF:   {p['pdf_url']}")
    if p.get("fields_of_study"):
        print(f"  领域:  {', '.join(p['fields_of_study'])}")
    print(f"  ── 评分 ──")
    print(f"  综合得分: {p['final_score']:.3f}")
    print(f"  相关性: {p['relevance_score']:.2f} | 权威性: {p['authority_score']:.2f} | "
          f"时效性: {p['recency_score']:.2f} | 引用: {p['citation_score']:.2f} | "
          f"多样性: {p['diversity_score']:.2f}")
    abstract = p.get("abstract", "")
    if len(abstract) > 250:
        abstract = abstract[:250] + "..."
    print(f"  ── 摘要 ──")
    print(f"  {abstract}")
    if p.get("judge_reasoning"):
        reasoning = p["judge_reasoning"]
        if len(reasoning) > 350:
            reasoning = reasoning[:350] + "..."
        print(f"  ── AI 评判 ──")
        print(f"  {reasoning}")

print(f"\n{'=' * 90}")
print(f"  共 {data['total_papers']} 篇论文  |  耗时 {data['latency_ms']:.0f} ms")
if data.get("metrics"):
    m = data["metrics"]
    print(f"  LLM 调用 {m.get('llm_calls', 0)} 次  |  Token 用量: {m.get('token_usage', 0)}")
    print(f"  数据源: {', '.join(m.get('search_sources_used', []))}")
print(f"{'=' * 90}")
