# UNIQUE.md — Deepfake Detector Architecture & Differentiators

This repository is a **video deepfake detector** (REAL vs FAKE classification). It does **not** synthesize or swap faces.

---

## 1. High-Level Architecture

```
Raw video (Celeb-DF / FF+ / custom)
        │
        ▼
┌───────────────────────┐
│  prepare_data.py      │  MTCNN face detect (conf ≥ 0.95)
│  (or prepare_FF+.py)  │  crop → 224×224 → JPEG frames
└───────────┬───────────┘
            │
            ▼
  frames/{train|val}/{real|fake}/{video_id}/
  + {train,val}_metadata.csv
            │
            ▼
┌───────────────────────┐
│  VideoViT             │  google/vit-base-patch16-224
│  train_vit.py (v1)    │  late fusion + mean pool (ViT frozen)
│  train_vit2.py (v2)   │  CLS+patch fuse + temporal attention
└───────────┬───────────┘
            │
            ▼
  data/models/best_model.pth
            │
            ▼
┌───────────────────────┐
│  predict.py           │  single video → REAL/FAKE + confidence
│  predict_all.py       │  batch val folders → CSV
└───────────────────────┘
```

### Components

| Layer | Role | Key files |
|-------|------|-----------|
| Prep | Face-centric frame extraction & stratified split | `data/prepare/prepare_data.py`, `prepare_FF+.py` |
| Model | Frame-wise ViT encoder + temporal aggregation | `data/train_vit.py`, `data/train_vit2.py` |
| Infer | Load checkpoint, sample frames, softmax | `data/predict.py`, `data/predict_all.py` |
| Utils | YOLO face crop, splits, CUDA check, Kaggle FF+ | `data/utils/*` |

There is **no HTTP API or frontend** in the core codebase today.

---

## 2. Data Flow

### Training path

1. Place videos under `data/videos/{real|fake}/<dataset_subdir>/`.
2. `prepare_data.py` reads every 5th frame (`SAMPLE_RATE=5`), runs **MTCNN**, keeps the highest-confidence face only if `prob ≥ 0.95`, resizes to **224×224**, caps at **60 frames/video**.
3. Videos are stratified 80/20 into `frames/train` and `frames/val`; metadata CSVs record `video_id`, `label` (0=real, 1=fake), `frames_dir`, `num_frames_extracted`.
4. `DeepfakeDataset` samples **T frames** per video (32 in v1, 8 in v2), applies ImageNet normalize (+ horizontal flip on train).
5. `VideoViT` emits 2-class logits; best validation accuracy checkpoint is saved to `data/models/best_model.pth`.

### Inference path (`predict.py`)

1. Open video with OpenCV; uniformly sample `NUM_FRAMES` via `np.linspace`.
2. Resize + ImageNet normalize each frame (full frame today — see caveat below).
3. Batch shape `(1, T, 3, 224, 224)` → model → softmax → `FAKE` if class 1 else `REAL`, plus confidence %.

### Known train/serve gap

Training frames are **MTCNN face crops**. `predict.py` currently feeds **full frames**. Any production frontend/API should align preprocessing (face crop before ViT) without changing the trained weights or model class logic.

### Model-class consistency

- `predict.py` imports `VideoViT` from **`train_vit`** (v1).
- `predict_all.py` imports `VideoViT` from **`train_vit2`** (v2).

Weights from one architecture are **not** interchangeable with the other.

---

## 3. Model Details

### Backbone

- Hugging Face **`google/vit-base-patch16-224`** (`ViTModel`, not the classification head alone).
- Input: RGB 224×224, ImageNet mean/std.

### VideoViT v1 (`train_vit.py`) — late fusion

- ViT **fully frozen**.
- Per-frame: `pooler_output` ([CLS]).
- Temporal: **mean pool** over frames → `Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→2)`.
- Config defaults: 32 frames, batch 8, AdamW `1e-5`, 20 epochs, CrossEntropy.

### VideoViT v2 (`train_vit2.py`) — attention fusion (primary differentiator)

- ViT starts frozen; **`unfreeze_vit()` after epoch 5** (`UNFREEZE_AFTER=5`).
- Per-frame embedding: **`CLS + mean(patch tokens)`**.
- Temporal: **L2-norm soft attention** over frames, then weighted sum (not mean pool).
- Head: same 256-dim projector + 2-way classifier.
- Loss: CrossEntropy with **`label_smoothing=0.1`**.
- Config defaults: 8 frames, batch 2; saves `best_model.pth` and `last_model.pth`.

---

## 4. What Makes This Pipeline Unique

| Differentiator | Why it matters |
|----------------|----------------|
| **Video-level ViT with late fusion** | Classifies the *sequence*, not a single still; aggregates temporal evidence of manipulation. |
| **CLS + patch-mean fusion (v2)** | Uses both global ([CLS]) and spatial patch context before temporal pooling — richer than CLS-only. |
| **Norm-based temporal attention (v2)** | Frames with stronger embedding magnitude get higher weight; avoids treating every frame equally. |
| **Staged fine-tuning (v2)** | Stable early training on a frozen backbone, then full ViT adaptation after epoch 5. |
| **High-confidence face gate (0.95)** | Training signal is restricted to high-quality face crops, reducing background noise and failed detections. |
| **Dual dataset tracks** | Celeb-DF-style MTCNN pipeline *and* FaceForensics+ 1-FPS path (`prepare_FF+.py`), plus DFDC_Frames / YOLO crop utilities. |
| **Dual trainers** | Simple frozen baseline (v1) vs improved attention + unfreeze schedule (v2) for ablation and iteration. |
| **Label smoothing (v2)** | Softens overconfidence on noisy deepfake labels common in public datasets. |

### What is *not* unique (standard practice)

Binary CE classification, ImageNet-pretrained ViT-B/16, face cropping for deepfake detection, Celeb-DF folder conventions.

---

## 5. Runtime & Path Conventions

- Almost all paths are prefixed `deepfake_detector/data/...`.
- Scripts expect the process **cwd to be the parent of the repo folder** (e.g. `Desktop`), not inside `deepfake_detector`.
- No CLI argparse: edit constants (`VIDEO_PATH`, `MODEL_PATH`, `Config`, `dataset_structure`) in-file.
- Checkpoints under `data/models/` are **gitignored** and not shipped; train or supply `best_model.pth` before inference.

---

## 6. Outputs

| Mode | Output |
|------|--------|
| `predict.py` | Console: `FAKE` or `REAL` + confidence % |
| `predict_all.py` | CSV: `video_name,true_label,predicted_label,confidence` |
| Training | Console metrics (accuracy, precision, recall, F1, ROC-AUC) + `.pth` checkpoints |

---

## 7. Scope Boundary for Productization

Any web UI should treat this stack as a **detection service**: upload media → run preprocess + `VideoViT` → show REAL/FAKE, confidence, and sampled-frame preview. Core training/inference math should stay in the existing Python modules; a thin API wrapper is the intended integration surface.
