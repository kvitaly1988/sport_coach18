from __future__ import annotations
import numpy as np

def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        raise ImportError(f"OpenCV недоступен: {e}")

def _pil_text(img_bgr, text, org, size=22):
    cv2 = _require_cv2()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    font = None
    for p in ["app/assets/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(p, size=size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=(255,255,255))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]
JOINT_NAMES_RU = {'elbow_left':'Левый локоть','elbow_right':'Правый локоть','shoulder_left':'Левое плечо','shoulder_right':'Правое плечо','hip_left':'Левое бедро','hip_right':'Правое бедро','knee_left':'Левое колено','knee_right':'Правое колено','ankle_left':'Левая лодыжка','ankle_right':'Правая лодыжка','torso':'Корпус (наклон)'}
IDX_MAP = {'elbow_left':13,'elbow_right':14,'shoulder_left':11,'shoulder_right':12,'hip_left':23,'hip_right':24,'knee_left':25,'knee_right':26,'ankle_left':27,'ankle_right':28}

def _valid_xy(xy, vis=None, vis_thresh=0.5):
    return (np.isfinite(xy[0]) and np.isfinite(xy[1]) and 0.0 <= float(xy[0]) <= 1.0 and 0.0 <= float(xy[1]) <= 1.0 and (vis is None or (np.isfinite(vis) and float(vis) >= vis_thresh)))

def _to_px(xy01, w, h):
    x = int(round(float(xy01[0]) * w))
    y = int(round(float(xy01[1]) * h))
    x = 0 if x < 0 else (w - 1 if x >= w else x)
    y = 0 if y < 0 else (h - 1 if y >= h else y)
    return (x, y)

def draw_skeleton(frame_bgr, lm_norm01, vis_thresh: float = 0.5):
    cv2 = _require_cv2()
    img = frame_bgr.copy()
    coords = np.asarray(lm_norm01, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 33 or coords.shape[1] < 2:
        return img
    xy = coords[:, :2]
    vis = coords[:, 2] if coords.shape[1] >= 3 else None
    h, w = img.shape[:2]
    for a, b in CONNECTIONS:
        va = None if vis is None else vis[a]
        vb = None if vis is None else vis[b]
        if _valid_xy(xy[a], va, vis_thresh) and _valid_xy(xy[b], vb, vis_thresh):
            cv2.line(img, _to_px(xy[a], w, h), _to_px(xy[b], w, h), (0,255,0), 2)
    for i in range(33):
        vi = None if vis is None else vis[i]
        if _valid_xy(xy[i], vi, vis_thresh):
            cv2.circle(img, _to_px(xy[i], w, h), 3, (0,0,255), -1)
    return img

def make_side_by_side(user_img, ref_img, is_bad: bool, err_value: float):
    cv2 = _require_cv2()
    h1, w1 = user_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    H = max(h1, h2)
    if h1 != H:
        user_img = cv2.resize(user_img, (int(w1 * H / h1), H))
    if h2 != H:
        ref_img = cv2.resize(ref_img, (int(w2 * H / h2), H))
    combo = np.concatenate([user_img, ref_img], axis=1)
    color = (0,0,255) if is_bad else (0,200,0)
    cv2.rectangle(combo, (0,0), (combo.shape[1]-1, combo.shape[0]-1), color, 3)
    combo = _pil_text(combo, f"Средняя ошибка: {err_value:.1f}°", (14, 12), size=22)
    combo = _pil_text(combo, "Пользователь", (14, H-30), size=20)
    combo = _pil_text(combo, "Эталон", (user_img.shape[1]+14, H-30), size=20)
    return combo

def draw_joint_overlay(user_frame_bgr, lm_user_norm01, lm_ref_norm01, joint_key: str):
    cv2 = _require_cv2()
    img = user_frame_bgr.copy()
    h, w = img.shape[:2]
    lu = np.asarray(lm_user_norm01, dtype=np.float32)
    lr = np.asarray(lm_ref_norm01, dtype=np.float32)
    if lu.ndim != 2 or lr.ndim != 2 or lu.shape[0] < 33 or lr.shape[0] < 33:
        return img
    if joint_key == "torso":
        u_sh = np.nanmean(np.stack([lu[11,:2], lu[12,:2]]), axis=0)
        r_sh = np.nanmean(np.stack([lr[11,:2], lr[12,:2]]), axis=0)
        if not (np.all(np.isfinite(u_sh)) and np.all(np.isfinite(r_sh))):
            return img
        pt_u = _to_px(u_sh, w, h)
        pt_r = _to_px(r_sh, w, h)
    else:
        idx = IDX_MAP.get(joint_key)
        if idx is None:
            return img
        if not (np.all(np.isfinite(lu[idx,:2])) and np.all(np.isfinite(lr[idx,:2]))):
            return img
        pt_u = _to_px(lu[idx,:2], w, h)
        pt_r = _to_px(lr[idx,:2], w, h)
    cv2.circle(img, pt_u, 7, (0,0,255), -1)
    cv2.drawMarker(img, pt_r, (0,255,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
    cv2.line(img, pt_u, pt_r, (0,255,255), 2)
    name = JOINT_NAMES_RU.get(joint_key, joint_key)
    img = _pil_text(img, name, (max(10, pt_u[0]+10), max(10, pt_u[1]-30)), size=22)
    return img
