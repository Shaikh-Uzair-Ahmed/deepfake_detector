import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import random
import numpy as np
from transformers import ViTModel, ViTConfig # Using ViTModel for features, not classification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Configuration ---
class Config:
    NUM_FRAMES_PER_VIDEO_INPUT = 32 # How many frames your model will actually see per video
    FACE_SIZE = (224, 224) # Ensure this matches the size used in prepare_data.py
    FRAME_OUTPUT_ROOT = 'deepfake_detector/data/frames' # Make sure this matches
    MODELS_OUTPUT_ROOT = 'deepfake_detector/data/models'

    BATCH_SIZE = 8 # Adjust based on your GPU memory
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20
    LOG_INTERVAL = 10 # Log training loss every N batches

    # ViT configuration
    VIT_MODEL_NAME = 'google/vit-base-patch16-224'
    NUM_CLASSES = 2 # Real/Fake

# --- Dataset Class ---
class DeepfakeDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform
        
        metadata_path = os.path.join(Config.FRAME_OUTPUT_ROOT, f'{split}_metadata.csv')
        self.metadata_df = pd.read_csv(metadata_path)

        # Filter out videos that didn't have enough frames extracted, if desired
        # self.metadata_df = self.metadata_df[self.metadata_df['num_frames_extracted'] >= Config.NUM_FRAMES_PER_VIDEO_INPUT]
        
        if len(self.metadata_df) == 0:
            raise ValueError(f"No videos found for {split} split after loading metadata. Check {metadata_path} and frame counts.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        record = self.metadata_df.iloc[idx]
        frames_dir = record['frames_dir']
        label = record['label']

        all_frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        # Handle cases where a video might have fewer frames than expected
        if len(all_frame_files) == 0: # If for some reason no frames were saved
             # Return a tensor of zeros and a dummy label, or handle error
             # For simplicity, we'll raise an error during training if this happens frequently
             print(f"Warning: No frames found for video {record['video_id']} at {frames_dir}. Returning dummy data.")
             dummy_frame = torch.zeros((3, *Config.FACE_SIZE))
             frames_tensor = torch.stack([dummy_frame] * Config.NUM_FRAMES_PER_VIDEO_INPUT)
             return frames_tensor, torch.tensor(label, dtype=torch.long) # Use long for CrossEntropyLoss

        if len(all_frame_files) < Config.NUM_FRAMES_PER_VIDEO_INPUT:
            # Pad by repeating frames (can also use a specific padding frame if desired)
            sampled_frame_files = random.choices(all_frame_files, k=Config.NUM_FRAMES_PER_VIDEO_INPUT)
        else:
            # Randomly sample N frames
            sampled_frame_files = random.sample(all_frame_files, k=Config.NUM_FRAMES_PER_VIDEO_INPUT)
        
        frames_tensor_list = []
        for frame_file in sampled_frame_files:
            img_path = os.path.join(frames_dir, frame_file)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames_tensor_list.append(image)
        
        # Stack all frame tensors: (NUM_FRAMES, C, H, W)
        frames_tensor = torch.stack(frames_tensor_list)

        return frames_tensor, torch.tensor(label, dtype=torch.long) # Use long for CrossEntropyLoss

# --- Model Definition (VideoViT for Late Fusion) ---
class VideoViT(nn.Module):
    def __init__(self, num_frames=Config.NUM_FRAMES_PER_VIDEO_INPUT, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.num_frames = num_frames

        # Load a pre-trained ViT model (without the classification head)
        vit_config = ViTConfig.from_pretrained(Config.VIT_MODEL_NAME)
        self.vit_encoder = ViTModel.from_pretrained(Config.VIT_MODEL_NAME, config=vit_config)
        
        # Freeze ViT layers initially (optional, but good for stability)
        # You can unfreeze and fine-tune later
        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        # Temporal aggregator: Mean pooling followed by linear layers
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(vit_config.hidden_size, 256), # Project ViT CLS token embedding
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classification head
        self.classifier = nn.Linear(256, num_classes) # For 2 classes (real/fake)

    def forward(self, frames_tensor):
        # frames_tensor shape: (batch_size, num_frames, C, H, W)
        batch_size, num_frames, C, H, W = frames_tensor.shape

        # Reshape for ViT: Treat each frame as a separate input initially
        # (batch_size * num_frames, C, H, W)
        frames_reshaped = frames_tensor.view(-1, C, H, W)

        # Pass through ViT encoder
        # vit_outputs.pooler_output gives the [CLS] token embedding for each frame
        vit_outputs = self.vit_encoder(pixel_values=frames_reshaped)
        
        frame_embeddings = vit_outputs.pooler_output 
        
        # Reshape back to (batch_size, num_frames, hidden_size)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)

        # Temporal aggregation (mean pooling example)
        aggregated_embedding = frame_embeddings.mean(dim=1) # (batch_size, hidden_size)

        # Pass through temporal aggregator
        aggregated_embedding = self.temporal_aggregator(aggregated_embedding)

        # Final classification
        logits = self.classifier(aggregated_embedding)
        return logits

# --- Training and Validation Function ---
def train_model():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize(Config.FACE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet Norm
    ])
    val_transform = transforms.Compose([
        transforms.Resize(Config.FACE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets and DataLoaders ---
    try:
        train_dataset = DeepfakeDataset(split='train', transform=train_transform)
        val_dataset = DeepfakeDataset(split='val', transform=val_transform)
    except ValueError as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure prepare_data.py has been run successfully and metadata files exist.")
        return

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2)

    print(f"Training on {len(train_dataset)} videos, validating on {len(val_dataset)} videos.")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model, Optimizer, Loss ---
    model = VideoViT(num_frames=Config.NUM_FRAMES_PER_VIDEO_INPUT, num_classes=Config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Use CrossEntropyLoss for 2 classes, labels as long

    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0

        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} (Training) ---")
        for batch_idx, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            frames = frames.to(device) # frames_tensor: (batch_size, num_frames, C, H, W)
            labels = labels.to(device) # labels: (batch_size,) type long

            optimizer.zero_grad()
            outputs = model(frames) # outputs will be logits: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train_predictions += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

            if batch_idx % Config.LOG_INTERVAL == 0 and batch_idx > 0:
                tqdm.write(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train_predictions / total_train_predictions
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        all_probs = [] # For ROC AUC
        
        print(f"--- Epoch {epoch+1}/{Config.NUM_EPOCHS} (Validation) ---")
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                frames = frames.to(device)
                labels = labels.to(device)

                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of being 'fake'
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, zero_division=0)
        val_recall = recall_score(all_labels, all_predictions, zero_division=0)
        val_f1 = f1_score(all_labels, all_predictions, zero_division=0)
        try:
            val_roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError: # Happens if only one class is present in batch
            val_roc_auc = 0.5 # Default to random if cannot calculate

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}, "
              f"ROC AUC: {val_roc_auc:.4f}")

        # --- Save best model ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(Config.MODELS_OUTPUT_ROOT, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path} with accuracy: {best_val_accuracy:.4f}")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    train_model()