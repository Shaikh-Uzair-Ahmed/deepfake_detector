import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from train_vit import VideoViT, Config  # import your model

# --- Configuration ---
MODEL_PATH = "deepfake_detector/data/models/best_model.pth"
FRAMES_DIR = r"deepfake_detector/data/frames/val"  # contains "real" and "fake"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(Config.FACE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Load trained model ---
model = VideoViT(num_frames=Config.NUM_FRAMES_PER_VIDEO_INPUT, num_classes=Config.NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_video(video_folder):
    """Predict one label for all frames in a single video folder."""
    frame_files = sorted([
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if len(frame_files) == 0:
        print(f"⚠️ No frames found in {video_folder}")
        return None

    # Take up to NUM_FRAMES_PER_VIDEO_INPUT frames
    selected_frames = frame_files[:Config.NUM_FRAMES_PER_VIDEO_INPUT]

    frames_tensor_list = []
    for frame_path in selected_frames:
        img = Image.open(frame_path).convert("RGB")
        frames_tensor_list.append(transform(img))

    # Stack into (1, num_frames, C, H, W)
    frames_tensor = torch.stack(frames_tensor_list).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(frames_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()

    return pred_label, confidence


# --- Loop through dataset ---
results = []

for label_folder in ["real", "fake"]:
    folder_path = os.path.join(FRAMES_DIR, label_folder)
    true_label = 0 if label_folder == "real" else 1

    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        if not os.path.isdir(video_path):
            continue

        result = predict_video(video_path)
        if result is None:
            continue

        pred_label, confidence = result
        results.append({
            "video_name": video_name,
            "true_label": label_folder,
            "predicted_label": "real" if pred_label == 0 else "fake",
            "confidence": confidence
        })

        print(f"🎥 {video_name}: Predicted → {'FAKE' if pred_label else 'REAL'} ({confidence*100:.2f}%)")

# --- Save to CSV ---
df = pd.DataFrame(results)
df.to_csv("val_video_predictions.csv", index=False)
print("\n✅ Predictions saved to val_video_predictions.csv")
