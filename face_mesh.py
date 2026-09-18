"""
face_mesh.py
------------
A thin wrapper around MediaPipe's face-landmark model.

Why this file exists: MediaPipe changed its Python API. Older versions expose
the classic `mp.solutions.face_mesh.FaceMesh`, while newer ones (0.10.22+)
removed it in favour of the Tasks API (`FaceLandmarker`). This wrapper tries
the legacy API first and falls back to the Tasks API, so the rest of the
project never has to care which version is installed.

Either way the interface is the same:

    fm = FaceMeshDetector()
    landmarks = fm.process(rgb_frame)     # list of 468+ points, or None

Each landmark has .x and .y in normalised 0..1 coordinates, exactly like
MediaPipe returns them.
"""

from __future__ import annotations

import os

TASK_MODEL_FILENAME = "face_landmarker.task"
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class FaceMeshDetector:
    """Detects 468 facial landmarks on a single face, on any MediaPipe version."""

    def __init__(self, model_path: str | None = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.backend = None
        self._impl = None
        self._timestamp_ms = 0

        import mediapipe as mp

        # ---------- backend 1: legacy solutions API ----------
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_mesh"):
            self._impl = solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.backend = "solutions"
            return

        # ---------- backend 2: Tasks API ----------
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", TASK_MODEL_FILENAME
            )
        if not os.path.exists(model_path):
            raise RuntimeError(
                "This MediaPipe version no longer ships the classic Face Mesh API, and "
                f"the Tasks model file was not found at:\n    {model_path}\n\n"
                "Fix it in either of these two ways:\n"
                "  1) pip install mediapipe==0.10.14        (recommended - classic API)\n"
                f"  2) download the model into assets/ :\n     {TASK_MODEL_URL}\n"
            )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._impl = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        self.backend = "tasks"

    # ------------------------------------------------------------------ #
    def process(self, rgb_frame):
        """Return the landmark list for the first detected face, or None."""
        if self.backend == "solutions":
            result = self._impl.process(rgb_frame)
            if not result.multi_face_landmarks:
                return None
            return result.multi_face_landmarks[0].landmark

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )
        self._timestamp_ms += 33
        result = self._impl.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    # ------------------------------------------------------------------ #
    def close(self):
        try:
            self._impl.close()
        except Exception:
            pass
