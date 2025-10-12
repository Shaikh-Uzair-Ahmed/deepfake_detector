import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from train_vit import VideoViT, Config  # Import the exact trained model class

# --- Configuration ---
MODEL_PATH = "deepfake_detector/data/models/best_model.pth"
VIDEO_PATH = "deepfake_detector/data/videos/fake/Celeb-synthesis/id0_id1_0005.mp4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing same as training ---
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

# --- Extract frames ---
def extract_frames(video_path, max_frames=32):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# --- Predict ---
def predict(video_path):
    frames = extract_frames(video_path, max_frames=Config.NUM_FRAMES_PER_VIDEO_INPUT)
    if len(frames) == 0:
        print("❌ No frames extracted. Check video path.")
        return

    # Preprocess frames
    frames_tensor_list = [transform(f) for f in frames]
    frames_tensor = torch.stack(frames_tensor_list).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        outputs = model(frames_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_label].item()

    label_str = "FAKE" if predicted_label == 1 else "REAL"
    print(f"🎥 Prediction for '{os.path.basename(video_path)}': {label_str} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    predict(VIDEO_PATH)
