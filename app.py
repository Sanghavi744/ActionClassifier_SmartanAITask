from flask import Flask, render_template, request
import os
import torch
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
import cv2
import random

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ['bicep_curl', 'lateral_raise', 'squat']

# Load model
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def load_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total < num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        start = random.randint(0, total - num_frames)
        indices = list(range(start, start + num_frames))

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    suggestions = []
    video_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        # Predict
        frames = load_video_frames(filename)
        tensor_frames = [transform(f) for f in frames]
        x = torch.stack(tensor_frames, dim=1).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=1).item()
        prediction = classes[pred]

        # Video for preview
        video_url = "/" + filename

        # Suggestions
        if prediction == "bicep_curl":
            suggestions = [
                ('Hammer Curl', 'https://youtu.be/zC3nLlEvin4?si=jjkoDoS5ycIgarli'),
                ('Concentration Curl', 'https://youtu.be/soxrZlIl35U?si=JwSzA8UUJCcV6lLq')
            ]
        elif prediction == "squat":
            suggestions = [
                ('Goblet Squat', 'https://www.youtube.com/watch?v=6xwGFn-J_QA'),
                ('Front Squat', 'https://www.youtube.com/watch?v=tlfGU6vH5Q0')
            ]
        elif prediction == "lateral_raise":
            suggestions = [
                ('Cable Lateral Raise', 'https://www.youtube.com/watch?v=cI3XkDo3k4k'),
                ('Dumbbell Upright Row', 'https://www.youtube.com/watch?v=jaOolG3Y0os')
            ]

    return render_template("index.html", prediction=prediction, video_url=video_url, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
