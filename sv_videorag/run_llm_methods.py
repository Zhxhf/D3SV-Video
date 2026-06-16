import json
import re
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from sv_videorag.run_doc_methods import (
    load_docs,
    get_docs_for_item,
    retrieve_docs,
    verify_evidence,
    refine_query,
)

LETTERS = ["A", "B", "C", "D", "E"]

def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))

def save_json(x, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(x, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def clean_pred(text):
    s = str(text).strip()
    m = re.search(r"\b([A-E])\b", s.upper())
    if m:
        return m.group(1)
    for ch in s.upper():
        if ch in LETTERS:
            return ch
    return "A"

class QwenTextAnswerer:
    def __init__(self, model_path):
        print("===== loading Qwen2-VL text answerer =====")
        print("model_path:", model_path)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        self.model.eval()

    @torch.inference_mode()
    def answer(self, question, options, evidence, max_new_tokens=8):
        ev_text = "\n".join(
            [f"[{i}] {d.get('timestamp','')}: {d.get('text','')}" for i, d in enumerate(evidence)]
        )
        opt_text = "\n".join(options)

        prompt = f"""You are answering a multiple-choice video question using retrieved video captions.

Question:
{question}

Options:
{opt_text}

Retrieved video evidence:
{ev_text}

Instruction:
Choose the best answer. Reply with only one letter from A, B, C, D, E. Do not explain.
"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt"
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return clean_pred(decoded), decoded

def select_docs_for_method(item, docs, method, topk):
    q = item.get("question", "")

    if method == "direct_doc":
        return docs[:topk]

    if method == "llovi_lite":
        # LLoVi-lite: use dense captions in original temporal order.
        return docs[:min(len(docs), max(topk, 12))]

    if method == "drvideo_lite":
        # DrVideo-lite: document retrieval with temporal/reasoning hints.
        query = q + " temporal event reason result before after"
        return retrieve_docs(docs, query, topk)

    if method == "videorag_lite":
        # Video-RAG-lite: retrieve visually aligned semantic cues.
        query = q + " object action scene OCR ASR speech person"
        return retrieve_docs(docs, query, topk)

    raise ValueError(method)

def run_baseline_llm(item, docs, method, topk, answerer):
    evidence = select_docs_for_method(item, docs, method, topk)
    pred, raw_pred = answerer.answer(item.get("question", ""), item.get("options", []), evidence)
    ans = item.get("answer", "")

    return {
        "uid": item.get("uid", ""),
        "dataset": item.get("dataset", ""),
        "method": method,
        "video_path": item.get("video", ""),
        "question": item.get("question", ""),
        "options": item.get("options", []),
        "answer": ans,
        "pred": pred,
        "correct": bool(ans and pred == ans),
        "evidence": evidence,
        "evidence_score": max([float(d.get("score", 0.0) or 0.0) for d in evidence], default=0.0),
        "selected_round": 0,
        "extra": {
            "raw_pred": raw_pred
        }
    }

def run_sv_llm(item, docs, topk, max_rounds, threshold, margin, answerer):
    q = item.get("question", "")
    options = item.get("options", [])

    pool = []
    query = q + " object action scene OCR ASR speech person"

    for r in range(max_rounds):
        evidence = retrieve_docs(docs, query, topk)
        pred, raw_pred = answerer.answer(q, options, evidence)

        score, qtype, req, parts = verify_evidence(q, pred, options, evidence)

        pool.append({
            "round": r,
            "pred": pred,
            "raw_pred": raw_pred,
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

    # conservative selection: only switch when evidence score is clearly better
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
    ap.add_argument("--docs_json", required=True)
    ap.add_argument("--method", required=True,
                    choices=["direct_doc", "llovi_lite", "drvideo_lite", "videorag_lite", "sv_videorag"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--margin", type=float, default=0.06)
    args = ap.parse_args()

    data = load_json(args.input)
    docs_map = load_docs(args.docs_json)
    answerer = QwenTextAnswerer(args.model_path)

    out = []
    doc_source_counter = Counter()

    for idx, item in enumerate(data):
        docs, doc_key = get_docs_for_item(item, docs_map)
        doc_source_counter[doc_key] += 1

        print(f"[{idx+1}/{len(data)}] method={args.method} uid={item.get('uid')} doc_key={doc_key} docs={len(docs)}")

        if args.method == "sv_videorag":
            y = run_sv_llm(
                item, docs,
                topk=args.topk,
                max_rounds=args.max_rounds,
                threshold=args.threshold,
                margin=args.margin,
                answerer=answerer
            )
        else:
            y = run_baseline_llm(item, docs, args.method, args.topk, answerer)

        y["extra"]["doc_key"] = doc_key
        y["extra"]["num_docs"] = len(docs)
        out.append(y)

        # incremental save to avoid losing progress
        if (idx + 1) % 10 == 0:
            save_json(out, args.output + ".partial")

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
