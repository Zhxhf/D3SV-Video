import json
import re
import argparse
from pathlib import Path
from collections import Counter

LETTERS = ["A", "B", "C", "D", "E"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(s):
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", str(s).lower())

def score_text(query, text):
    q = Counter(tokenize(query))
    t = Counter(tokenize(text))
    if not q or not t:
        return 0.0
    hit = sum(min(q[w], t[w]) for w in q)
    return hit / (sum(q.values()) + 1e-6)

def option_letter(text):
    s = str(text).strip()
    for ch in s:
        if ch.upper() in LETTERS:
            return ch.upper()
    return ""

def heuristic_answer(question, options, evidence_text=""):
    """
    这是 smoke test 用的弱规则，不是论文模型。
    真正跑实验时，这里要替换成 Qwen2-VL / VideoMind 的生成结果。
    """
    query = question + " " + evidence_text
    best_i, best_s = 0, -1
    for i, opt in enumerate(options):
        s = score_text(query, opt)
        if s > best_s:
            best_i, best_s = i, s
    return LETTERS[best_i] if options and best_i < len(LETTERS) else "A"

def build_pseudo_docs(item):
    """
    当前服务器没有完整视频，所以先用 question/options 构造伪文档做流程测试。
    后面有 caption/video 后，替换成 clip captions 即可。
    """
    docs = []
    q = item.get("question", "")
    for i, opt in enumerate(item.get("options", [])):
        docs.append({
            "clip_id": i,
            "timestamp": f"pseudo-{i}",
            "text": f"Question-related candidate evidence: {q} {opt}",
            "score": 0.0
        })
    return docs

def retrieve_docs(item, topk=4, mode="caption_rag"):
    docs = build_pseudo_docs(item)
    q = item.get("question", "")
    for d in docs:
        if mode == "direct":
            d["score"] = 0.0
        elif mode == "llovi_lite":
            d["score"] = score_text(q, d["text"])
        elif mode == "drvideo_lite":
            d["score"] = score_text(q + " temporal event reason result", d["text"])
        elif mode == "videorag_lite":
            d["score"] = score_text(q + " object action scene OCR ASR", d["text"])
        else:
            d["score"] = score_text(q, d["text"])
    docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    return docs[:topk]

def run_method(data, method, topk):
    outputs = []
    for item in data:
        if method == "direct":
            evd = []
            evidence_text = ""
        else:
            evd = retrieve_docs(item, topk=topk, mode=method)
            evidence_text = "\n".join(d["text"] for d in evd)
        pred = heuristic_answer(item.get("question", ""), item.get("options", []), evidence_text)
        ans = item.get("answer", "")
        outputs.append({
            "uid": item.get("uid", ""),
            "dataset": item.get("dataset", ""),
            "method": method,
            "video_path": item.get("video", ""),
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": ans,
            "pred": pred,
            "correct": bool(ans and pred == ans),
            "evidence": evd,
            "evidence_score": max([d["score"] for d in evd], default=0.0),
            "selected_round": 0,
            "extra": {"note": "smoke-test rule baseline; replace generator with LVLM for real experiment"}
        })
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--method", required=True, choices=["direct", "llovi_lite", "drvideo_lite", "videorag_lite"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    data = load_json(args.input)
    out = run_method(data, args.method, args.topk)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n = len(out)
    c = sum(x["correct"] for x in out)
    print("method:", args.method)
    print("saved:", args.output)
    print("n:", n, "correct:", c, "acc:", round(c / max(n, 1) * 100, 2))

if __name__ == "__main__":
    main()
