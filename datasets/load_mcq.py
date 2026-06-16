import json
import csv
import argparse
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E"]

def load_json_or_csv(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)
    if isinstance(x, dict):
        if "data" in x and isinstance(x["data"], list):
            return x["data"]
        return list(x.values())
    return x

def pick(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in [None, ""]:
            return row[k]
    return default

def normalize_options(row):
    opts = pick(row, ["options", "choices", "candidates", "answer_candidates"], [])

    if isinstance(opts, dict):
        out = []
        for l in LETTERS:
            if l in opts:
                out.append(f"{l}. {opts[l]}")
        if out:
            return out

    if isinstance(opts, list):
        out = []
        for i, o in enumerate(opts):
            s = str(o).strip()
            if len(s) >= 2 and s[0].upper() in LETTERS and s[1] in [".", ")", "、"]:
                out.append(s)
            else:
                out.append(f"{LETTERS[i]}. {s}" if i < len(LETTERS) else s)
        if out:
            return out

    # MLVU / common A-E
    out = []
    for i, k in enumerate(["A", "B", "C", "D", "E"]):
        if k in row:
            out.append(f"{k}. {row[k]}")
        elif f"option_{k}" in row:
            out.append(f"{k}. {row[f'option_{k}']}")
    if out:
        return out

    # IntentQA / NextQA style: a0-a4
    out = []
    for i in range(5):
        k = f"a{i}"
        if k in row and str(row[k]).strip() != "":
            out.append(f"{LETTERS[i]}. {row[k]}")
    return out

def normalize_answer(ans):
    s = str(ans).strip()
    if s == "":
        return ""

    # IntentQA: answer=0/1/2/3/4
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]

    # MLVU: answer=A/B/C/D or text containing letter
    for ch in s:
        if ch.upper() in LETTERS:
            return ch.upper()

    return s

def normalize_rows(rows, dataset="unknown", limit=-1):
    out = []
    for idx, r in enumerate(rows):
        uid = pick(r, ["uid", "id", "qid", "question_id", "sample_id"], str(idx))
        q = pick(r, ["question", "query", "Q"], "")
        opts = normalize_options(r)
        ans = normalize_answer(pick(r, ["answer", "ans", "label", "gt", "correct_answer"], ""))
        video = pick(r, ["video", "video_path", "video_id", "vid", "video_name"], "")
        duration = pick(r, ["duration", "video_duration"], "")
        item = {
            "uid": str(uid),
            "dataset": dataset,
            "video": str(video),
            "duration": duration,
            "question": str(q),
            "options": opts,
            "answer": ans,
            "raw": r,
        }
        out.append(item)
        if limit > 0 and len(out) >= limit:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", default="unknown")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    rows = load_json_or_csv(args.input)
    rows = normalize_rows(rows, args.dataset, args.limit)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.output, "n=", len(rows))
    if rows:
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])

if __name__ == "__main__":
    main()
