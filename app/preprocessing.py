import numpy as np
from scipy.signal import savgol_filter

def normalize_landmarks(seq):
    out = []
    for lm in seq:
        lm = np.asarray(lm, dtype=np.float32)
        if lm.shape[0] < 33 or lm.shape[1] < 2 or np.isnan(lm[:, :2]).all():
            out.append(lm)
            continue

        hip_l, hip_r = lm[23, :2], lm[24, :2]
        if np.any(np.isnan(hip_l)) or np.any(np.isnan(hip_r)):
            center = np.nanmean(lm[:, :2], axis=0)
        else:
            center = (hip_l + hip_r) / 2.0

        sh_l, sh_r = lm[11, :2], lm[12, :2]
        if np.any(np.isnan(sh_l)) or np.any(np.isnan(sh_r)):
            scale = np.nanstd(lm[:, :2])
        else:
            sh = (sh_l + sh_r) / 2.0
            scale = float(np.linalg.norm(sh - center))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.nanstd(lm[:, :2]) or 1.0)

        n = lm.copy()
        n[:, :2] = (n[:, :2] - center) / scale
        out.append(n)
    return out

def _angle(a, b, c):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return np.nan
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def compute_angles_sequence(seq_norm):
    keys = [
        "elbow_left","elbow_right","shoulder_left","shoulder_right",
        "hip_left","hip_right","knee_left","knee_right","ankle_left","ankle_right","torso"
    ]
    out = {k: [] for k in keys}
    for lm in seq_norm:
        lm = np.asarray(lm, dtype=np.float32)
        if lm.shape[0] < 33:
            for k in keys:
                out[k].append(np.nan)
            continue
        xy = lm[:, :2]
        out["elbow_left"].append(_angle(xy[11], xy[13], xy[15]))
        out["elbow_right"].append(_angle(xy[12], xy[14], xy[16]))
        out["shoulder_left"].append(_angle(xy[13], xy[11], xy[23]))
        out["shoulder_right"].append(_angle(xy[14], xy[12], xy[24]))
        out["hip_left"].append(_angle(xy[11], xy[23], xy[25]))
        out["hip_right"].append(_angle(xy[12], xy[24], xy[26]))
        out["knee_left"].append(_angle(xy[23], xy[25], xy[27]))
        out["knee_right"].append(_angle(xy[24], xy[26], xy[28]))
        out["ankle_left"].append(_angle(xy[25], xy[27], xy[31]))
        out["ankle_right"].append(_angle(xy[26], xy[28], xy[32]))

        hip_mid = np.nanmean(np.stack([xy[23], xy[24]]), axis=0)
        sh_mid = np.nanmean(np.stack([xy[11], xy[12]]), axis=0)
        if np.any(np.isnan(hip_mid)) or np.any(np.isnan(sh_mid)):
            out["torso"].append(np.nan)
        else:
            v = sh_mid - hip_mid
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                out["torso"].append(np.nan)
            else:
                vert = np.array([0.0, -1.0], dtype=np.float32)
                cosang = float(np.dot(v / nv, vert))
                cosang = max(-1.0, min(1.0, cosang))
                out["torso"].append(float(np.degrees(np.arccos(cosang))))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}

def smooth_series(series, window=11, poly=3):
    out = {}
    for k, v in series.items():
        v = np.asarray(v, dtype=np.float32)
        if len(v) >= window and np.isfinite(v).sum() >= window:
            vv = v.copy()
            m = np.isfinite(vv)
            if not m.all():
                idx = np.where(m)[0]
                if len(idx) == 0:
                    out[k] = v
                    continue
                first = idx[0]
                vv[:first] = vv[first]
                for i in range(first + 1, len(vv)):
                    if not np.isfinite(vv[i]):
                        vv[i] = vv[i - 1]
            try:
                out[k] = savgol_filter(vv, window, poly).astype(np.float32)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out
