# DrVideo: Document Retrieval Based Long Video Understanding

> **CVPR 2025** — Code for the paper *“DrVideo: Document Retrieval Based Long Video Understanding”*.

Most of the existing methods for video understanding primarily focus on videos only lasting tens of seconds, with limited exploration of techniques for handling long videos. The increased number of frames in long videos poses two main challenges: difficulty in locating key information and performing long-range reasoning. Thus, we propose DrVideo, a document-retrieval-based system designed for long video understanding. Our key idea is to convert the long-video understanding problem into a long-document understanding task so as to effectively leverage the power of large language models. Specifically, DrVideo first transforms a long video into a coarse text-based long document to initially retrieve key frames and then updates the documents with the augmented key frame information. It then employs an agent-based iterative loop to continuously search for missing information and augment the document until sufficient question-related information is gathered for making the final predictions in a chain-of-thought manner. Extensive experiments on long video benchmarks confirm the effectiveness of our method. DrVideo significantly outperforms existing LLM-based state-of-the-art methods on EgoSchema benchmark (3 minutes), MovieChat-1K benchmark (10 minutes), and the long split of Video-MME benchmark (average of 44 minutes).

---

## Repository Structure
```
.
├── main.py                 # End-to-end pipeline entry (builds doc + answers QA)
├── eval.py                 # Metrics for MCQ-style evaluation
├── dataset.py              # EgoSchema-style dataset loader
├── model.py                # LLM wrapper (OpenAI + fallbacks)
├── prompts.py              # Prompt templates for doc building and reasoning
├── models/
│   └── blip2_model.py      # Image captioning (LLaVA-Next / BLIP)
├── utils/                  # Feature extraction, segmentation, etc.
├── util.py                 # Arg parser, IO helpers
├── data/egoschema/         # Example JSONs (durations, annos)
└── requirements.txt        # Tested package versions
```

---

## Installation
> We recommend a **conda** environment with CUDA. The provided requirements were tested with **torch 1.9.0 + cu111** and **detectron2 0.6**.

```bash
conda create -n drvideo python=3.9 -y
conda activate drvideo

# PyTorch (adjust CUDA as needed)
pip install --extra-index-url https://download.pytorch.org/whl   torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0

# Core deps
pip install -r requirements.txt
```

---

## Data Preparation
The default dataset interface follows **EgoSchema-style** multiple-choice QA.

Place (or symlink) files under `data/egoschema/`:
- `lavila_subset_add_time.json` — narrations / descriptions per `uid`
- `subset_anno.json` — question & options per `uid` (with optional `truth`)
- `duration.json` — video duration (seconds) per `uid`

Your videos are expected at:
```
/mnt/2030154A30152874/videos/videos/{uid}.mp4
```
Change this path in `dataset.py` if your storage layout differs.

---

## Quick Start
Run the **end-to-end** pipeline (document building → answer prediction → evaluation). The arguments below appear in `main.py`, but some (e.g., `--image_caption`, `--dense_caption`, `--image_captioner_device`, `--feature_extractor`) may require additional implementation to be functional in your environment.

```bash
python main.py   --dataset egoschema   --data_path data/egoschema/lavila_subset_add_time.json   --anno_path data/egoschema/subset_anno.json   --duration_path data/egoschema/duration.json   --fps 1.0   --num_examples_to_run -1     # Output
  --output_base_path output/   --output_filename egoschema_run.json     # LLM & prompting
  --model gpt-4-turbo   --temperature 0.0   --prompt_type qa_standard   --openai_api_key "$OPENAI_API_KEY"
```

---

## Evaluation Only
If you already have a predictions JSON (same schema the pipeline writes), you can compute MCQ accuracy:

```bash
python -c "from eval import eval_qa_egoschema; import json;  data=json.load(open('output/egoschema_run.json'));  acc=eval_qa_egoschema(data); print(acc)"
```

---

## Logging & Outputs
- Intermediate and final artifacts are written under `--output_base_path`.
- The main output file contains per-UID entries:
  ```json
  {
    "<uid>": {
      "pred": 0–4,            // model choice (A–E)
      "pred_text": "A–E",     // textual label
      "rationale": "...",      // (if enabled) model rationale
      "evidence": {...}        // captions / retrieved snippets
    },
    ...
  }
  ```

---

## Troubleshooting
- **CUDA OOM**: switch to `--image_captioner blip` or set device to CPU.
- **Detectron2 install**: use the wheel URL pinned in `requirements.txt`.
- **SpaCy model**: if `en_core_web_sm` install fails, re-run the exact URL in `requirements.txt`.
- **OpenAI errors**: ensure the `--openai_api_key` is valid.

---

## Citation
If you find this repo useful, please cite:
```
@InProceedings{Ma_2025_CVPR,
    author    = {Ma, Ziyu and Gou, Chenhui and Shi, Hengcan and Sun, Bin and Li, Shutao and Rezatofighi, Hamid and Cai, Jianfei},
    title     = {DrVideo: Document Retrieval Based Long Video Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {18936-18946}
}
```

---

## License
This code is released for research purposes. Please check each model's own license (OpenAI, LLaVA-Next, BLIP, etc.).
