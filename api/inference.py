"""Inference wrapper around train_vit2.VideoViT — does not modify the model class."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_CANDIDATES = [
    DATA_DIR / "models" / "best_model.pth",
    DATA_DIR / "models" / "last_model.pth",
]

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from train_vit2 import Config, VideoViT  # noqa: E402


def resolve_checkpoint() -> Path:
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No VideoViT checkpoint found. Expected "
        f"{MODEL_CANDIDATES[0]} (train with data/train_vit2.py first)."
    )


class Detector:
    """VideoViT v2 detector with MTCNN face crops aligned to training preprocess."""

    MIN_FACE_CONFIDENCE = 0.95
    CANDIDATE_FRAMES = 24

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mtcnn = None
        self.checkpoint_path = None
        self.error = None
        self.lock = threading.Lock()
        self.transform = transforms.Compose(
            [
                transforms.Resize(Config.FACE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.num_frames = Config.NUM_FRAMES_PER_VIDEO_INPUT

    def load(self) -> None:
        try:
            self.checkpoint_path = resolve_checkpoint()
            model = VideoViT(
                num_frames=self.num_frames,
                num_classes=Config.NUM_CLASSES,
            ).to(self.device)
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.model = model

            from facenet_pytorch import MTCNN

            self.mtcnn = MTCNN(
                image_size=Config.FACE_SIZE[0],
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
            )
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            raise

    @property
    def ready(self) -> bool:
        return self.model is not None and self.error is None

    def info(self) -> dict:
        return {
            "loaded": self.ready,
            "device": str(self.device),
            "model": "VideoViT v2",
            "backbone": Config.VIT_MODEL_NAME,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "num_frames": self.num_frames,
            "error": self.error,
        }

    def crop_face(self, image: Image.Image) -> Image.Image | None:
        if self.mtcnn is None:
            return None
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None
        best = int(np.argmax(probs))
        if float(probs[best]) < self.MIN_FACE_CONFIDENCE:
            return None
        x1, y1, x2, y2 = [int(v) for v in boxes[best]]
        arr = np.array(image)
        h, w = arr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        face = Image.fromarray(arr[y1:y2, x1:x2])
        return face.resize(Config.FACE_SIZE, Image.Resampling.BILINEAR)

    def _select_frames(self, faces: list[Image.Image], fallbacks: list[Image.Image]) -> tuple[list[Image.Image], bool]:
        source = faces if faces else fallbacks
        used_face_crop = bool(faces)
        if not source:
            return [], used_face_crop
        if len(source) >= self.num_frames:
            indices = np.linspace(0, len(source) - 1, self.num_frames).astype(int)
            return [source[i] for i in indices], used_face_crop
        padded = list(source)
        while len(padded) < self.num_frames:
            padded.append(source[-1])
        return padded, used_face_crop

    def frames_from_image(self, path: str) -> tuple[list[Image.Image], bool]:
        image = Image.open(path).convert("RGB")
        face = self.crop_face(image)
        fallback = image.resize(Config.FACE_SIZE, Image.Resampling.BILINEAR)
        return self._select_frames(
            [face] if face is not None else [],
            [fallback],
        )

    def frames_from_video(self, path: str) -> tuple[list[Image.Image], bool]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            sequential = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                sequential.append(frame)
            cap.release()
            if not sequential:
                raise RuntimeError("No frames could be read from the video.")
            total = len(sequential)
            n_cand = min(self.CANDIDATE_FRAMES, total)
            indices = np.linspace(0, total - 1, n_cand).astype(int)
            bgr_frames = [sequential[i] for i in indices]
        else:
            n_cand = min(self.CANDIDATE_FRAMES, total)
            indices = np.linspace(0, total - 1, n_cand).astype(int)
            bgr_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    bgr_frames.append(frame)
            cap.release()

        faces: list[Image.Image] = []
        fallbacks: list[Image.Image] = []
        for frame in bgr_frames:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fallbacks.append(rgb.resize(Config.FACE_SIZE, Image.Resampling.BILINEAR))
            face = self.crop_face(rgb)
            if face is not None:
                faces.append(face)

        selected, used_face_crop = self._select_frames(faces, fallbacks)
        if not selected:
            raise RuntimeError("No usable frames extracted from the video.")
        return selected, used_face_crop

    def predict(self, media_path: str, preview_dir: str, on_status=None) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not loaded.")

        ext = os.path.splitext(media_path)[1].lower()
        with self.lock:
            if on_status:
                on_status("extracting_frames", "Sampling frames and detecting faces…")
            if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                frames, used_face_crop = self.frames_from_image(media_path)
            else:
                frames, used_face_crop = self.frames_from_video(media_path)

            os.makedirs(preview_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(os.path.join(preview_dir, f"{i}.jpg"), quality=92)

            if on_status:
                on_status("detecting", "Running VideoViT v2…")
            tensors = [self.transform(f.convert("RGB")) for f in frames]
            batch = torch.stack(tensors).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
                confidence = float(probs[pred].item())
                real_p = float(probs[0].item())
                fake_p = float(probs[1].item())

        return {
            "label": "FAKE" if pred == 1 else "REAL",
            "confidence": confidence,
            "real_probability": real_p,
            "fake_probability": fake_p,
            "num_frames": len(frames),
            "used_face_crop": used_face_crop,
        }


detector = Detector()
