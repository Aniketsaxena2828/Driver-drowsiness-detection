# Driver Drowsiness & Distraction Detection System

A real-time computer vision system that monitors a driver through a webcam or recorded video and detects:

- 😴 Drowsiness
- 🥱 Yawning
- 👀 Distraction / looking away from the road

Built using **Python, OpenCV and MediaPipe**. No custom training or dataset is required.

## Features

- Real-time drowsiness detection using **Eye Aspect Ratio (EAR)**
- Yawning detection using **Mouth Aspect Ratio (MAR)**
- Head pose estimation using `cv2.solvePnP`
- Consecutive-frame detection to reduce false alerts
- Real-time EAR, MAR, head pose and FPS display
- Audio alarm system
- Webcam and video-file support
- Annotated video output
- CSV event logging
- Configurable detection thresholds
- Self-test without requiring a webcam

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Computer Vision

## How It Works

```text
Webcam / Video
      ↓
MediaPipe Face Detection
      ↓
Facial Landmarks
      ↓
 ┌───────────────┬───────────────┬───────────────┐
 ↓               ↓               ↓
EAR Detection   MAR Detection   Head Pose
 ↓               ↓               ↓
Drowsiness       Yawning         Distraction
 └───────────────┴───────────────┴───────────────┘
                    ↓
              State Machine
                    ↓
              Alert + Logging
