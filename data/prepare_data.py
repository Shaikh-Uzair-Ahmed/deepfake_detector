import cv2
import os
import torch
from facenet_pytorch import MTCNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# --- Configuration ---
# Update VIDEO_ROOT to point to the 'videos' directory
VIDEO_ROOT = 'deepfake_detector/data/videos' 
FRAME_OUTPUT_ROOT = 'deepfake_detector/data/frames'
FACE_SIZE = (224, 224) # Standard size for ViT input
SAMPLE_RATE = 5 # Take every 5th frame to reduce processing/storage
MAX_FRAMES_PER_VIDEO = 60 # Cap frames to prevent excessively long sequences
TRAIN_SPLIT_RATIO = 0.8
MIN_FACE_CONFIDENCE = 0.95 # Only save frames with high confidence face detection

# --- Initialize MTCNN for face detection ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for MTCNN: {device}")

mtcnn = MTCNN(
    image_size=FACE_SIZE[0],
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=device
)

# --- Helper function to process a single video ---
def process_video(video_path, output_dir, video_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frame_count = 0
    saved_frames_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % SAMPLE_RATE == 0:
            rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            boxes, probs = mtcnn.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:
                best_face_idx = np.argmax(probs)
                if probs[best_face_idx] < MIN_FACE_CONFIDENCE:
                    frame_count += 1
                    continue

                box = boxes[best_face_idx]
                x1, y1, x2, y2 = [int(b) for b in box]

                face_img = frame[y1:y2, x1:x2]
                
                if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    face_img = cv2.resize(face_img, FACE_SIZE)

                    frame_filename = f"{video_id}_frame_{frame_count:04d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, face_img)
                    saved_frames_paths.append(frame_path)
                    
                    if len(saved_frames_paths) >= MAX_FRAMES_PER_VIDEO:
                        break

        frame_count += 1
    cap.release()
    return saved_frames_paths

# --- Main script for data preparation ---
def prepare_data():
    # 1. Create necessary output directories
    # These will be created as 'deepfake_detector/data/frames/train/real/Celeb-real_id0_0000/' etc.
    os.makedirs(FRAME_OUTPUT_ROOT, exist_ok=True) # Ensure base frames dir exists

    all_video_paths = []
    all_video_labels = [] # 0 for real, 1 for fake
    all_video_ids = []    # Unique ID for each video (e.g., 'Celeb-real_id0_0000')

    # Collect all video paths and labels
    # Define the expected subdirectories
    dataset_structure = {
        'real': ['Celeb-real', 'Youtube-real'],
        'fake': ['Celeb-synthesis']
    }

    print("Scanning video directories...")
    for label_type, subdirs in dataset_structure.items():
        for subdir in subdirs:
            current_video_dir = os.path.join(VIDEO_ROOT, label_type, subdir)
            if not os.path.exists(current_video_dir):
                print(f"Warning: Directory {current_video_dir} not found. Skipping.")
                continue

            for video_filename in os.listdir(current_video_dir):
                if video_filename.endswith(('.mp4', '.avi', '.mov', '.webm', '.flv')):
                    video_full_path = os.path.join(current_video_dir, video_filename)
                    # Create a unique video_id by combining subdir and filename (without extension)
                    unique_video_id = f"{subdir}_{os.path.splitext(video_filename)[0]}"

                    all_video_paths.append(video_full_path)
                    all_video_labels.append(0 if label_type == 'real' else 1)
                    all_video_ids.append(unique_video_id)

    if not all_video_paths:
        print("No videos found in the specified nested directories. Please check your VIDEO_ROOT path and dataset structure.")
        return

    # Split video IDs, paths, and labels into train and validation sets
    train_video_ids, val_video_ids, \
    train_video_paths, val_video_paths, \
    train_labels, val_labels = train_test_split(
        all_video_ids, all_video_paths, all_video_labels,
        test_size=(1 - TRAIN_SPLIT_RATIO), stratify=all_video_labels, random_state=42
    )

    metadata_records_train = []
    metadata_records_val = []

    print("\n--- Processing training videos ---")
    for i in tqdm(range(len(train_video_paths)), desc="Training Videos"):
        video_id = train_video_ids[i]
        video_path = train_video_paths[i]
        label = train_labels[i]
        
        # Determine the target output directory for frames based on split and label
        output_sub_dir_base = os.path.join(FRAME_OUTPUT_ROOT, 'train', 'real' if label == 0 else 'fake')
        output_video_frames_dir = os.path.join(output_sub_dir_base, video_id) # Unique folder for each video's frames
        os.makedirs(output_video_frames_dir, exist_ok=True)
        
        saved_frames_paths = process_video(video_path, output_video_frames_dir, video_id)
        if saved_frames_paths:
            metadata_records_train.append({
                'video_id': video_id,
                'label': label,
                'frames_dir': output_video_frames_dir, # Path to the directory of extracted frames
                'num_frames_extracted': len(saved_frames_paths)
            })

    print("\n--- Processing validation videos ---")
    for i in tqdm(range(len(val_video_paths)), desc="Validation Videos"):
        video_id = val_video_ids[i]
        video_path = val_video_paths[i]
        label = val_labels[i]

        output_sub_dir_base = os.path.join(FRAME_OUTPUT_ROOT, 'val', 'real' if label == 0 else 'fake')
        output_video_frames_dir = os.path.join(output_sub_dir_base, video_id) # Unique folder for each video's frames
        os.makedirs(output_video_frames_dir, exist_ok=True)
        
        saved_frames_paths = process_video(video_path, output_video_frames_dir, video_id)
        if saved_frames_paths:
            metadata_records_val.append({
                'video_id': video_id,
                'label': label,
                'frames_dir': output_video_frames_dir,
                'num_frames_extracted': len(saved_frames_paths)
            })
            
    # Save metadata
    pd.DataFrame(metadata_records_train).to_csv(os.path.join(FRAME_OUTPUT_ROOT, 'train_metadata.csv'), index=False)
    pd.DataFrame(metadata_records_val).to_csv(os.path.join(FRAME_OUTPUT_ROOT, 'val_metadata.csv'), index=False)
    print(f"\nMetadata saved to {os.path.join(FRAME_OUTPUT_ROOT, 'train_metadata.csv')} and {os.path.join(FRAME_OUTPUT_ROOT, 'val_metadata.csv')}")

if __name__ == "__main__":
    prepare_data()