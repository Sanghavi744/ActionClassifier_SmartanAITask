# app.py — Flask Video Action Classifier

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
import cv2
import random

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define classes
classes = ['bicep_curl', 'lateral_raise', 'squat']
similar_exercises = {
    "bicep_curl": ["hammer curl", "concentration curl", "preacher curl"],
    "lateral_raise": ["front raise", "dumbbell shoulder press", "arnold press"],
    "squat": ["goblet squat", "sumo squat", "bulgarian split squat"]
}
# 👇 Similar exercises suggestion map
similar_exercises = {
    'bicep_curl': [
        ("Hammer Curl", "https://www.youtube.com/watch?v=zC3nLlEvin4"),
        ("Concentration Curl", "https://youtu.be/Jvj2wV0vOYU?si=2ulvn5f7wpPgwwf2")
    ],
    'lateral_raise': [
        ("Front Raise", "https://youtube.com/shorts/yHx8wPv4RPo?si=Km18IdTsICcUGpDO"),
        ("Bent-Over Lateral Raise", "https://www.youtube.com/watch?v=SWjzFaH9QXA")
    ],
    'squat': [
        ("Goblet Squat", "https://www.youtube.com/watch?v=MeIiIdhvXT4"),
        ("Bulgarian Split Squat", "https://www.youtube.com/watch?v=2C-uNgKwPLE")
    ]
}


# Load model
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# Video frame extractor
def load_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
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

# Prediction
def predict(path):
    frames = load_video_frames(path)
    if not frames:
        return None, []
    tensor_frames = [transform(f) for f in frames]
    x = torch.stack(tensor_frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()
    label = classes[pred]
    suggestions = similar_exercises.get(label, [])
    return label, suggestions

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    suggestions = []
    show_result = False

    if request.method == "POST":
        video = request.files["file"]
        if video:
            filename = secure_filename(video.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            video.save(filepath)

            prediction, suggestions = predict(filepath)
            show_result = True

    return render_template("index.html", prediction=prediction, suggestions=suggestions, show_result=show_result)

if __name__ == "__main__":
    app.run(debug=True)
