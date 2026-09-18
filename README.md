# Driver Drowsiness & Distraction Detection System

Real-time computer vision system that watches a driver through a webcam (or a recorded video) and raises an alert when it detects **drowsiness** (prolonged eye closure), **yawning**, or **distraction** (head turned away from the road).

Built with Python, OpenCV and MediaPipe Face Mesh. No training, no dataset, no GPU — it runs on any laptop webcam in real time.

---

## Features

- **Drowsiness detection** — Eye Aspect Ratio (EAR) across 6 eye landmarks per eye
- **Yawn detection** — Mouth Aspect Ratio (MAR) across the mouth landmarks
- **Distraction detection** — head yaw / pitch / roll via `cv2.solvePnP`
- **Consecutive-frame state machine** — alerts only fire after a condition holds for N frames, so normal blinking never triggers a false alarm
- **Live overlay** — EAR / MAR / yaw / pitch, counters, running event totals and FPS
- **Audio alarm** — non-blocking, with automatic fallbacks on every OS
- **Saved outputs** — annotated `.mp4` video + timestamped `.csv` event log
- **Full CLI** — every threshold adjustable from the terminal via `argparse`
- **Self-test** — `python test_detector.py` verifies the whole pipeline without a camera

---

## Quick start

```bash
# 1. create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. verify everything works (no webcam needed)
python test_detector.py

# 4. run it
python main.py
```

---

## Usage

```bash
# live webcam with window
python main.py

# webcam + save annotated video and event log
python main.py --source webcam --output outputs/annotated_output.mp4 --log outputs/events_log.csv

# process a recorded video file
python main.py --source drive.mp4 --output outputs/annotated_output.mp4 --log outputs/events_log.csv

# more sensitive eyes, stricter head-turn limit
python main.py --ear-threshold 0.22 --ear-frames 25 --yaw-threshold 20

# headless / server run, no window and no sound
python main.py --source drive.mp4 --no-display --no-sound
```

Press **`q`** or **ESC** in the window to quit.

### All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `webcam` | `webcam`, a camera index (`0`, `1`), or a video file path |
| `--output` | – | path to save the annotated video |
| `--log` | – | path to save the CSV event log |
| `--ear-threshold` | `0.25` | EAR below this = eye counted as closed |
| `--ear-frames` | `20` | consecutive closed frames before a DROWSINESS alert |
| `--mar-threshold` | `0.60` | MAR above this = mouth counted as wide open |
| `--mar-frames` | `15` | consecutive open frames before a YAWN alert |
| `--yaw-threshold` | `25.0` | degrees of left/right head turn allowed |
| `--pitch-threshold` | `20.0` | degrees of up/down head tilt allowed |
| `--pose-frames` | `20` | consecutive looking-away frames before a DISTRACTION alert |
| `--alarm` | `assets/alarm.wav` | alert sound file |
| `--no-sound` | off | disable the audio alarm |
| `--no-display` | off | do not open a window |
| `--no-landmarks` | off | do not draw the landmark dots |
| `--max-frames` | `0` | stop after N frames (0 = unlimited) |

---

## How it works

```
   Webcam / Video Frame
            |
            v
   1. Face Mesh (MediaPipe)  ->  468 facial landmarks
            |
            +--> 2a. EAR  (eye landmarks)    -> eyes open or closed?
            |
            +--> 2b. MAR  (mouth landmarks)  -> yawning?
            |
            +--> 2c. Head pose (solvePnP)    -> looking away?
            |
            v
   3. State machine (consecutive-frame counters + thresholds)
            |
            v
   4. Alert: on-screen banner + audio alarm + overlays
            |
            v
   5. Save annotated video (cv2.VideoWriter) + CSV event log
```

### Eye Aspect Ratio (EAR)

Six landmarks are taken around each eye (`p1`–`p6`, `p1`/`p4` being the left and right corners):

```
EAR = ( |p2-p6| + |p3-p5| ) / ( 2 * |p1-p4| )
```

The numerator is the eye height (two vertical distances), the denominator the eye width. An open eye sits around **0.25–0.30**; when the eye closes the vertical distances collapse toward zero while the width stays constant, so EAR drops sharply toward **0**. It is a pure geometric ratio — no machine learning classifier involved.

### Mouth Aspect Ratio (MAR)

Exactly the same formula applied to the mouth landmarks: vertical mouth opening over mouth width. A sustained high MAR is a yawn.

### Head Pose Estimation

Six stable 2D landmarks (nose tip, chin, both outer eye corners, both mouth corners) are matched against a generic 3D face model using `cv2.solvePnP`. It answers: *given these known 3D points and where they land in the 2D image, what head rotation would produce this view?* The resulting rotation vector becomes a rotation matrix (`cv2.Rodrigues`) and is then decomposed (`cv2.RQDecomp3x3`) into:

- **Yaw** — head turned left / right
- **Pitch** — head tilted up / down
- **Roll** — head tilted sideways

If yaw or pitch stays past its threshold for enough consecutive frames, the driver is flagged as distracted.

### The state machine — the key design decision

For each of the three checks the detector keeps a counter:

- **increments** every frame the condition is true
- **resets to zero** the moment the condition becomes false
- **fires an alert** only once the counter crosses its threshold

A normal blink lasts 100–400 ms — roughly 3–12 frames at webcam FPS. Alerting on a single low-EAR frame would fire on every blink. Requiring the condition to hold for ~20 consecutive frames (≈0.6–0.8 s) filters blinks out naturally while still catching genuine prolonged eye closure almost immediately. Each alert episode is also latched, so one long closure logs exactly one event instead of hundreds.

---

## Project structure

```
driver-drowsiness-detection/
├── main.py              # CLI entry point (argparse, video loop, writer, summary)
├── detector.py          # DrowsinessDetector class: EAR, MAR, head pose, state machine
├── face_mesh.py         # MediaPipe wrapper (classic Face Mesh API + Tasks API fallback)
├── utils.py             # landmark/geometry helpers, overlays, AlarmPlayer, EventLogger
├── test_detector.py     # self-test, runs without a webcam
├── requirements.txt
├── assets/
│   └── alarm.wav        # alert sound
├── outputs/             # annotated_output.mp4 and events_log.csv land here
├── README.md
├── LICENSE
└── .gitignore
```

---

## Output

**Annotated video** — the original footage with the metrics panel, landmark dots, alert banners and FPS burned in.

**CSV event log** (`outputs/events_log.csv`):

```csv
timestamp,video_time_sec,frame,event,details
2026-09-18 13:14:41,2.63,79,DROWSINESS,EAR=0.020
2026-09-18 13:14:42,4.47,134,YAWN,MAR=0.800
```

**Terminal summary** printed when the run ends:

```
==============================================
 SESSION SUMMARY
==============================================
 frames processed   : 180
 duration           : 2.4s  (74.4 fps)
 drowsiness alerts  : 1
 yawns detected     : 1
 distraction alerts : 0
==============================================
```

---

## Tuning tips

- Getting false drowsiness alerts? **Lower** `--ear-threshold` to `0.21–0.23` (some people have naturally narrow eyes) or **raise** `--ear-frames`.
- Missing real drowsiness? **Raise** `--ear-threshold` or **lower** `--ear-frames`.
- Yawns not detected? **Lower** `--mar-threshold` to around `0.5`.
- Distraction firing while you glance at the mirror? **Raise** `--pose-frames`.
- Watch the live EAR/MAR readout while acting out each state — that is the fastest way to pick thresholds for your own face and camera.

---

## Troubleshooting

**`Could not open video source: webcam`** — another app is using the camera, or the index is wrong. Try `--source 1`.

**MediaPipe error about the classic Face Mesh API** — you installed a MediaPipe version ≥ 0.10.22, which removed it. Either run `pip install mediapipe==0.10.14`, or download `face_landmarker.task` from Google's model page into `assets/` and the Tasks API fallback in `face_mesh.py` takes over automatically.

**No sound** — install `playsound` (`pip install playsound==1.2.2`), or simply ignore it: the on-screen alert always works, and `--no-sound` disables audio entirely.

**Low FPS** — use a smaller camera resolution, or run with `--no-landmarks`.

---

## Viva notes

- The most important idea is the **consecutive-frame state machine** — it is what separates a blink from drowsiness.
- EAR and MAR are **geometric ratios**, not trained models, which is why they are fast and fully explainable.
- `solvePnP` solves the **Perspective-n-Point** problem: recovering 3D rotation and translation from known 3D points and their 2D projections.
- MediaPipe is used instead of the classic `dlib` 68-point predictor purely because it pip-installs with no build tools. **The math is identical** — only the source of the landmarks changes.
- Thresholds are per-person and per-camera; that is why they are exposed as CLI flags rather than hard-coded.

---

## License

MIT — see [LICENSE](LICENSE).
#   D r i v e r - d r o w s i n e s s - d e t e c t i o n  
 