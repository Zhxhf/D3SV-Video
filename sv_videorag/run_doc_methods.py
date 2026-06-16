import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

LETTERS = ["A", "B", "C", "D", "E"]

def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))

def save_json(x, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(x, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def tokenize(s):
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", str(s).lower())

def score_text(query, text):
    q = Counter(tokenize(query))
    t = Counter(tokenize(text))
    if not q or not t:
        return 0.0
    hit = sum(min(q[w], t[w]) for w in q)
    return hit / (sum(q.values()) + 1e-6)

def normalize_uid(x):
    return str(x).strip()

def load_docs(docs_json):
    """
    Expected docs format:
    [
      {
        "uid": "sample_or_video_id",
        "docs": [
          {"clip_id":0, "timestamp":"00:00-00:08", "text":"...", "modality":"caption"}
        ]
      }
    ]

    Also supports dict:
    {
      "video_id": ["caption1", "caption2"]
    }
    """
    if not docs_json:
        return {}
    p = Path(docs_json)
    if not p.exists():
        print("docs_json not found, fallback to pseudo docs:", docs_json)
        return {}

    x = load_json(p)
    out = {}

    if isinstance(x, list):
        for item in x:
            uid = normalize_uid(item.get("uid", item.get("video", item.get("video_id", ""))))
            docs = item.get("docs", [])
            if uid:
                out[uid] = normalize_docs(docs)
        return out

    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, list):
                docs = []
                for i, text in enumerate(v):
                    if isinstance(text, dict):
                        docs.append(text)
                    else:
                        docs.append({
                            "clip_id": i,
                            "timestamp": f"clip-{i}",
                            "text": str(text),
                            "modality": "caption"
                        })
                out[normalize_uid(k)] = normalize_docs(docs)
            elif isinstance(v, dict):
                docs = v.get("docs", v.get("captions", v.get("sentences", [])))
                out[normalize_uid(k)] = normalize_docs(docs)
            else:
                out[normalize_uid(k)] = normalize_docs([{
                    "clip_id": 0,
                    "timestamp": "unknown",
                    "text": str(v),
                    "modality": "caption"
                }])
        return out

    return {}

def normalize_docs(docs):
    out = []
    for i, d in enumerate(docs):
        if isinstance(d, str):
            out.append({
                "clip_id": i,
                "timestamp": f"clip-{i}",
                "text": d,
                "modality": "caption",
                "score": 0.0
            })
        elif isinstance(d, dict):
            out.append({
                "clip_id": d.get("clip_id", i),
                "timestamp": d.get("timestamp", d.get("time", f"clip-{i}")),
                "text": str(d.get("text", d.get("caption", d.get("sentence", d.get("description", ""))))),
                "modality": d.get("modality", "caption"),
                "score": float(d.get("score", 0.0) or 0.0)
            })
    return [d for d in out if d["text"].strip()]

def get_item_uid_candidates(item):
    raw = item.get("raw", {}) or {}
    cands = [
        item.get("uid", ""),
        item.get("video", ""),
        raw.get("video_path", ""),
        raw.get("video", ""),
        raw.get("video_id", ""),
        raw.get("vid", ""),
        raw.get("_zip_inner_path", ""),
    ]
    # 再加 basename
    more = []
    for c in cands:
        s = str(c)
        if "/" in s:
            more.append(Path(s).name)
            more.append(Path(s).stem)
        if "." in s:
            more.append(Path(s).stem)
    return [normalize_uid(x) for x in cands + more if str(x).strip()]

def build_pseudo_docs(item):
    docs = []
    q = item.get("question", "")
    for i, opt in enumerate(item.get("options", [])):
        docs.append({
            "clip_id": i,
            "timestamp": f"pseudo-{i}",
            "text": f"Pseudo evidence for pipeline test. Question: {q}. Candidate: {opt}",
            "modality": "pseudo",
            "score": 0.0
        })
    return docs

def get_docs_for_item(item, docs_map):
    for uid in get_item_uid_candidates(item):
        if uid in docs_map:
            return docs_map[uid], uid
    return build_pseudo_docs(item), "pseudo"

def infer_question_type(q):
    ql = q.lower()
    if any(w in ql for w in ["why", "reason", "cause", "because", "为什么", "原因"]):
        return "causal"
    if any(w in ql for w in ["after", "before", "then", "next", "first", "last", "之前", "之后", "然后"]):
        return "temporal"
    if any(w in ql for w in ["who", "person", "man", "woman", "boy", "girl", "谁"]):
        return "entity"
    if any(w in ql for w in ["what", "doing", "happen", "activity", "action", "什么", "发生"]):
        return "action"
    return "general"

def retrieve_docs(docs, query, topk=4):
    out = []
    for d in docs:
        x = dict(d)
        x["score"] = score_text(query, x.get("text", ""))
        out.append(x)
    return sorted(out, key=lambda z: z["score"], reverse=True)[:topk]

def heuristic_answer(question, options, evidence_text):
    """
    这里仍然是轻量答案器。真正论文实验可替换成 Qwen2-VL / VideoMind 生成。
    但是所有方法会使用同一个答案器，区别只来自证据检索与验证。
    """
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
            break

    qtype = infer_question_type(q)
    if qtype == "causal":
        req = f"Need causal evidence before the result, supporting option {pred}: {opt_text}"
    elif qtype == "temporal":
        req = f"Need temporal order evidence, supporting option {pred}: {opt_text}"
    elif qtype == "entity":
        req = f"Need same-entity and action evidence, supporting option {pred}: {opt_text}"
    elif qtype == "action":
        req = f"Need key action/event evidence, supporting option {pred}: {opt_text}"
    else:
        req = f"Need sufficient visual semantic evidence, supporting option {pred}: {opt_text}"
    return qtype, req

def verify_evidence(q, pred, options, evidence):
    qtype, req = evidence_requirement(q, pred, options)
    evidence_text = " ".join(d.get("text", "") for d in evidence)

    s_sem = score_text(q + " " + pred, evidence_text)
    s_req = score_text(req, evidence_text)

    if qtype == "causal":
        type_hint = "cause reason before because result"
    elif qtype == "temporal":
        type_hint = "before after then next first last temporal order"
    elif qtype == "entity":
        type_hint = "person object entity same action"
    elif qtype == "action":
        type_hint = "action event movement activity interaction"
    else:
        type_hint = "scene action object visual evidence"

    s_type = score_text(type_hint, evidence_text)
    s_evd = 0.4 * s_sem + 0.4 * s_req + 0.2 * s_type

    return s_evd, qtype, req, {
        "s_sem": s_sem,
        "s_req": s_req,
        "s_type": s_type
    }

def refine_query(q, qtype, req):
    if qtype == "causal":
        return q + " cause reason before previous event " + req
    if qtype == "temporal":
        return q + " before after then next temporal order " + req
    if qtype == "entity":
        return q + " same person object entity action " + req
    if qtype == "action":
        return q + " action event activity interaction " + req
    return q + " supporting visual evidence " + req

def run_baseline(item, docs, method, topk):
    q = item.get("question", "")
    options = item.get("options", [])

    if method == "direct_doc":
        selected = docs[:topk]
        ev_text = "\n".join(d.get("text", "") for d in selected)
    elif method == "llovi_lite":
        selected = docs[:]
        ev_text = "\n".join(d.get("text", "") for d in selected)
        selected = selected[:topk]
    elif method == "drvideo_lite":
        query = q + " temporal event reason result"
        selected = retrieve_docs(docs, query, topk)
        ev_text = "\n".join(d.get("text", "") for d in selected)
    elif method == "videorag_lite":
        query = q + " object action scene OCR ASR speech"
        selected = retrieve_docs(docs, query, topk)
        ev_text = "\n".join(d.get("text", "") for d in selected)
    else:
        raise ValueError(method)

    pred = heuristic_answer(q, options, ev_text)
    ans = item.get("answer", "")

    return {
        "uid": item.get("uid", ""),
        "dataset": item.get("dataset", ""),
        "method": method,
        "video_path": item.get("video", ""),
        "question": q,
        "options": options,
        "answer": ans,
        "pred": pred,
        "correct": bool(ans and pred == ans),
        "evidence": selected,
        "evidence_score": max([d.get("score", 0.0) for d in selected], default=0.0),
        "selected_round": 0,
        "extra": {}
    }

def run_sv(item, docs, topk, max_rounds, threshold, margin):
    q = item.get("question", "")
    options = item.get("options", [])

    pool = []
    query = q + " object action scene OCR ASR speech"

    for r in range(max_rounds):
        evidence = retrieve_docs(docs, query, topk)
        ev_text = "\n".join(d.get("text", "") for d in evidence)
        pred = heuristic_answer(q, options, ev_text)

        score, qtype, req, parts = verify_evidence(q, pred, options, evidence)

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
            "pool": pool
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--method", required=True,
                    choices=["direct_doc", "llovi_lite", "drvideo_lite", "videorag_lite", "sv_videorag"])
    ap.add_argument("--docs_json", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--margin", type=float, default=0.06)
    args = ap.parse_args()

    data = load_json(args.input)
    docs_map = load_docs(args.docs_json)

    out = []
    doc_source_counter = Counter()

    for item in data:
        docs, doc_key = get_docs_for_item(item, docs_map)
        doc_source_counter[doc_key] += 1

        if args.method == "sv_videorag":
            y = run_sv(item, docs, args.topk, args.max_rounds, args.threshold, args.margin)
        else:
            y = run_baseline(item, docs, args.method, args.topk)

        y["extra"]["doc_key"] = doc_key
        y["extra"]["num_docs"] = len(docs)
        out.append(y)

    save_json(out, args.output)

    n = len(out)
    c = sum(bool(x.get("correct")) for x in out)
    print("method:", args.method)
    print("saved:", args.output)
    print("n:", n, "correct:", c, "acc:", round(c / max(n, 1) * 100, 2))
    print("doc_source_top:", doc_source_counter.most_common(10))
    print("selected_round:", Counter(str(x.get("selected_round", 0)) for x in out))

if __name__ == "__main__":
    main()
