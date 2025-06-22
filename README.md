# 🏋️ Gym Action Classifier

This project implements a video-based action recognition system that classifies gym workout videos into one of three categories: **Bicep Curl**, **Lateral Raise**, or **Squat**. It uses a pre-trained 3D CNN model (R3D-18) and provides a simple Flask web interface for video upload and prediction.

## 📁 Dataset Used

We used a curated dataset of gym exercises:
- **Source**: [Kaggle - Gym Workout Exercises Video Dataset](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video)
- **Structure**: Videos are organized into folders per class.
- Each video is 3–5 seconds long and contains at least 16 frames.

## 🛠️ Preprocessing & Model

- **Preprocessing**:
  - Extract 16 frames from each video using OpenCV.
  - Resize frames to (112, 112).
  - Apply standard tensor transformation for input to 3D CNN.
  
- **Model**:
  - Pretrained [R3D-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html) from `torchvision`.
  - Final layer replaced with 3-class output head.
  - Trained using Adam optimizer and cross-entropy loss for 3 epochs (can be extended).

## 🧪 How to Run Training

1. Open `train.py` (provided separately).
2. Modify `data_path` to point to your training dataset.
3. Run in Colab or VSCode:

```bash
python train.py
