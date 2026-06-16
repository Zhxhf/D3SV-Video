import json
import re
import argparse
from pathlib import Path
from collections import Counter

LETTERS = ["A", "B", "C", "D", "E"]

def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))

def tokenize(s):
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", str(s).lower())

def score_text(query, text):
    q = Counter(tokenize(query))
    t = Counter(tokenize(text))
    if not q or not t:
        return 0.0
    hit = sum(min(q[w], t[w]) for w in q)
    return hit / (sum(q.values()) + 1e-6)

def infer_question_type(q):
    ql = q.lower()
    if any(w in ql for w in ["why", "reason", "cause", "为什么", "原因"]):
        return "causal"
    if any(w in ql for w in ["after", "before", "then", "next", "first", "最后", "之前", "之后"]):
        return "temporal"
    if any(w in ql for w in ["who", "person", "man", "woman", "boy", "girl", "谁"]):
        return "entity"
    if any(w in ql for w in ["what", "doing", "happen", "什么", "发生"]):
        return "action"
    return "general"

def build_pseudo_docs(item):
    docs = []
    q = item.get("question", "")
    for i, opt in enumerate(item.get("options", [])):
        docs.append({
            "clip_id": i,
            "timestamp": f"pseudo-{i}",
            "text": f"Visual semantic evidence candidate: {q} {opt}",
            "score": 0.0
        })
    return docs

def retrieve(item, query, topk):
    docs = build_pseudo_docs(item)
    for d in docs:
        d["score"] = score_text(query, d["text"])
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:topk]

def heuristic_answer(question, options, evidence_text):
    query = question + " " + evidence_text
    best_i, best_s = 0, -1
    for i, opt in enumerate(options):
        s = score_text(query, opt)
        if s > best_s:
            best_i, best_s = i, s
    return LETTERS[best_i] if options and best_i < len(LETTERS) else "A"

def evidence_requirement(q, pred, options):
    opt_text = ""
    for opt in options:
        if str(opt).strip().upper().startswith(pred):
            opt_text = opt
    qtype = infer_question_type(q)
    if qtype == "causal":
        req = f"Need evidence about the cause before the answer: {opt_text}"
    elif qtype == "temporal":
        req = f"Need evidence about temporal order before/after events supporting: {opt_text}"
    elif qtype == "entity":
        req = f"Need evidence that the same entity performs the action in: {opt_text}"
    elif qtype == "action":
        req = f"Need evidence about the key action/event: {opt_text}"
    else:
        req = f"Need sufficient visual evidence supporting: {opt_text}"
    return qtype, req

def verify(q, pred, options, evidence):
    qtype, req = evidence_requirement(q, pred, options)
    evidence_text = " ".join(d["text"] for d in evidence)

    s_sem = score_text(q + " " + pred, evidence_text)
    s_req = score_text(req, evidence_text)

    if qtype == "causal":
        s_type = score_text("cause reason before why", evidence_text)
    elif qtype == "temporal":
        s_type = score_text("before after then next temporal order", evidence_text)
    elif qtype == "entity":
        s_type = score_text("person object entity same action", evidence_text)
    else:
        s_type = score_text("action event scene visual", evidence_text)

    s_evd = 0.4 * s_sem + 0.4 * s_req + 0.2 * s_type
    return s_evd, qtype, req, {"s_sem": s_sem, "s_req": s_req, "s_type": s_type}

def refine_query(q, qtype, req):
    if qtype == "causal":
        return q + " reason cause before previous event " + req
    if qtype == "temporal":
        return q + " before after next then temporal order " + req
    if qtype == "entity":
        return q + " same person same object entity action " + req
    if qtype == "action":
        return q + " action event movement interaction " + req
    return q + " supporting evidence " + req

def run_one(item, topk=4, max_rounds=3, threshold=0.62, margin=0.06):
    q = item.get("question", "")
    options = item.get("options", [])

    pool = []
    query = q + " object action scene OCR ASR"
    for r in range(max_rounds):
        evidence = retrieve(item, query, topk)
        ev_text = "\n".join(d["text"] for d in evidence)
        pred = heuristic_answer(q, options, ev_text)
        score, qtype, req, parts = verify(q, pred, options, evidence)

        pool.append({
            "round": r,
            "pred": pred,
            "evidence": evidence,
            "score": score,
            "qtype": qtype,
            "requirement": req,
            "parts": parts,
            "query": query
        })

        if score >= threshold:
            break
        query = refine_query(q, qtype, req)

    # Conservative selection:
    # 不是盲目用最后一轮，而是分数高于旧答案 margin 才换。
    best = pool[0]
    for cand in pool[1:]:
        if cand["score"] > best["score"] + margin:
            best = cand

    ans = item.get("answer", "")
    return {
        "uid": item.get("uid", ""),
        "dataset": item.get("dataset", ""),
        "method": "sv_videorag",
        "video_path": item.get("video", ""),
        "question": q,
        "options": options,
        "answer": ans,
        "pred": best["pred"],
        "correct": bool(ans and best["pred"] == ans),
        "evidence": best["evidence"],
        "evidence_score": best["score"],
        "selected_round": best["round"],
        "extra": {
            "pool": pool,
            "note": "SV-VideoRAG smoke framework; replace pseudo docs with real clip captions/video evidence for real experiment"
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--margin", type=float, default=0.06)
    args = ap.parse_args()

    data = load_json(args.input)
    out = [run_one(x, args.topk, args.max_rounds, args.threshold, args.margin) for x in data]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n = len(out)
    c = sum(x["correct"] for x in out)
    print("method: sv_videorag")
    print("saved:", args.output)
    print("n:", n, "correct:", c, "acc:", round(c / max(n, 1) * 100, 2))
    print("selected_round:", Counter(str(x["selected_round"]) for x in out))

if __name__ == "__main__":
    main()
