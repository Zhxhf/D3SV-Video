# Unified Output Format

Each method should output a JSON list.

Example:

{
  "uid": "sample_id",
  "dataset": "mlvu",
  "method": "sv_videorag",
  "video_path": "xxx.mp4",
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B",
  "pred": "C",
  "correct": false,
  "evidence": [
    {
      "clip_id": 3,
      "timestamp": "00:30-00:45",
      "text": "...",
      "score": 0.78
    }
  ],
  "evidence_score": 0.72,
  "selected_round": 1,
  "extra": {}
}
