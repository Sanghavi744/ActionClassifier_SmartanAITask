import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

# ✅ Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Class labels
classes = ['bicep_curl', 'lateral_raise', 'squat']

# ✅ Load the pre-trained + fine-tuned model
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("model.pth", map_location=device))  # Ensure model.pth is in the same folder
model = model.to(device)
model.eval()

# ✅ Transform frames
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# ✅ Load and sample frames from a video
def load_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total < num_frames:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        start = random.randint(0, total - num_frames)
        indices = np.arange(start, start + num_frames)

    idx_set = set(indices)
    i = 0
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        i += 1
    cap.release()
    return frames

# ✅ Predict function
def predict(video_path):
    frames = load_video_frames(video_path)
    if not frames:
        print("❌ Error: Could not extract frames from video.")
        return

    tensor_frames = [transform(f) for f in frames]
    x = torch.stack(tensor_frames, dim=1).unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy().flatten()
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx] * 100

        print(f"✅ Prediction: {pred_class.upper()} ({confidence:.2f}% confidence)")

# ✅ Replace with your local test video path
if __name__ == "__main__":
    video_path = "/Users/sanghavikirkole/Documents/assignment_dataset/verified_data/verified_data/data_crawl_10s/squat/5aa83733-033b-4e53-8332-17a2bfaee7f9.mp4"  # Example: change this to your video path
    if os.path.exists(video_path):
        predict(video_path)
    else:
        print("❌ Provided video path does not exist.")
  