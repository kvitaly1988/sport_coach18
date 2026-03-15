from __future__ import annotations
import numpy as np

def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        raise ImportError(
            "Не удалось импортировать OpenCV. Для Streamlit Cloud нужны "
            "opencv-python-headless, packages.txt с libgl1 и Python 3.11. "
            f"Ошибка: {e}"
        )

def _require_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise ImportError(
            "Не удалось импортировать MediaPipe. Проверьте requirements.txt и runtime.txt. "
            f"Ошибка: {e}"
        )

def get_video_frames(path: str, max_width: int = 960):
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            continue
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frames.append(frame)
    cap.release()
    return float(fps), frames

def extract_pose_from_video(path: str, max_width: int = 960):
    cv2 = _require_cv2()
    mp = _require_mediapipe()
    fps, frames = get_video_frames(path, max_width=max_width)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            arr = np.array(
                [[p.x, p.y, getattr(p, "visibility", 1.0)] for p in res.pose_landmarks.landmark],
                dtype=np.float32
            )
        else:
            arr = np.full((33, 3), np.nan, np.float32)
        landmarks.append(arr)
    pose.close()
    return {"fps": fps, "frames": frames, "landmarks": landmarks}
