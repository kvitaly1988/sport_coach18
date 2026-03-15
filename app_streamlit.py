# -*- coding: utf-8 -*-
import os
import json
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st

from app.pose_extractor import extract_pose_from_video
from app.preprocessing import normalize_landmarks, compute_angles_sequence, smooth_series
from app.dtw_utils import stack_features, align_by_dtw
from app.visualization import draw_skeleton, make_side_by_side, draw_joint_overlay, JOINT_NAMES_RU
from app.recommendations import generate_ai_recommendations

CONFIG_PATH = Path("app/elements_config.json")
REF_DIR = Path("references")
REF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI-коуч", layout="wide")
st.title("🤸 AI-коуч: сравнение с эталоном")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def md5_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def analyze_cached(user_hash: str, user_path: str, ref_path: str, ref_mtime: float):
    user = extract_pose_from_video(user_path)
    ref = extract_pose_from_video(ref_path)

    user_norm = normalize_landmarks(user["landmarks"])
    ref_norm = normalize_landmarks(ref["landmarks"])
    user_angles = smooth_series(compute_angles_sequence(user_norm), window=11, poly=3)
    ref_angles = smooth_series(compute_angles_sequence(ref_norm), window=11, poly=3)

    feat_keys = [k for k in user_angles.keys() if any(s in k for s in ["torso", "hip", "knee"])] or list(user_angles.keys())[:6]
    uf = stack_features(user_angles, feat_keys)
    rf = stack_features(ref_angles, feat_keys)

    _, _, path = align_by_dtw(uf, rf)
    idx_user = [p[0] for p in path]
    idx_ref = [p[1] for p in path]

    angle_mae = {}
    common = set(user_angles.keys()).intersection(ref_angles.keys())
    for k in common:
        angle_mae[k] = float(np.nanmean(np.abs(user_angles[k][idx_user] - ref_angles[k][idx_ref])))

    dur_u = (idx_user[-1] - idx_user[0]) / max(1e-6, user["fps"])
    dur_r = (idx_ref[-1] - idx_ref[0]) / max(1e-6, ref["fps"])
    tempo_err = float(abs(dur_u - dur_r))

    return {
        "user_fps": float(user["fps"]),
        "ref_fps": float(ref["fps"]),
        "user_frames": user["frames"],
        "ref_frames": ref["frames"],
        "user_landmarks": user["landmarks"],
        "ref_landmarks": ref["landmarks"],
        "user_angles": user_angles,
        "ref_angles": ref_angles,
        "idx_user": idx_user,
        "idx_ref": idx_ref,
        "angle_mae": angle_mae,
        "tempo_err": tempo_err,
    }

def compute_score(angle_mae: dict, tempo_err: float, important: dict):
    if not angle_mae:
        return 0.0
    group_map = {
        "hip": ["hip_left", "hip_right"],
        "knee": ["knee_left", "knee_right"],
        "ankle": ["ankle_left", "ankle_right"],
        "shoulder": ["shoulder_left", "shoulder_right"],
        "elbow": ["elbow_left", "elbow_right"],
        "torso": ["torso"],
    }
    num = 0.0
    den = 0.0
    for g, keys in group_map.items():
        vals = [angle_mae.get(k, np.nan) for k in keys]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        err = float(np.mean(vals))
        w = float(important.get(g, 1.0))
        num += w * err
        den += w
    mean_werr = num / den if den > 0 else float(np.mean(list(angle_mae.values())))
    score = 100.0 - 2.5 * mean_werr - 10.0 * tempo_err
    return float(max(0.0, min(100.0, score)))

def group_from_tip(t: str):
    low = t.lower()
    if "таз" in low or "бедр" in low:
        return "hip"
    if "колен" in low:
        return "knee"
    if "лодыж" in low or "стоп" in low:
        return "ankle"
    if "корпус" in low:
        return "torso"
    if "плеч" in low:
        return "shoulder"
    if "локт" in low or "рук" in low:
        return "elbow"
    return None

def representative_joint_key(group: str):
    return {
        "hip": "hip_left",
        "knee": "knee_left",
        "ankle": "ankle_left",
        "shoulder": "shoulder_left",
        "elbow": "elbow_left",
        "torso": "torso",
    }.get(group)

def _require_cv2():
    import cv2
    return cv2

def build_sync_video_file(analysis_obj: dict, user_hash: str, pause_threshold: float = 12.0):
    cv2 = _require_cv2()
    A = analysis_obj
    aligned_len = len(A["idx_user"])
    keys = list(A["angle_mae"].keys())

    per_frame_err = np.zeros(aligned_len, dtype=np.float32)
    for i in range(aligned_len):
        s = 0.0
        c = 0
        for k in keys:
            u = A["user_angles"][k][A["idx_user"]][i]
            r = A["ref_angles"][k][A["idx_ref"]][i]
            if np.isfinite(u) and np.isfinite(r):
                s += abs(float(u) - float(r))
                c += 1
        per_frame_err[i] = (s / c) if c else 0.0

    out_path = os.path.join(tempfile.gettempdir(), f"sync_compare_{user_hash}.mp4")
    fu0 = max(0, min(A["idx_user"][0], len(A["user_frames"]) - 1))
    fr0 = max(0, min(A["idx_ref"][0], len(A["ref_frames"]) - 1))
    uf0 = draw_skeleton(A["user_frames"][fu0].copy(), A["user_landmarks"][fu0])
    rf0 = draw_skeleton(A["ref_frames"][fr0].copy(), A["ref_landmarks"][fr0])
    first = make_side_by_side(uf0, rf0, per_frame_err[0] >= pause_threshold, float(per_frame_err[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать видеофайл сравнения.")
    writer.write(first)

    for i in range(1, aligned_len):
        fu = max(0, min(A["idx_user"][i], len(A["user_frames"]) - 1))
        fr = max(0, min(A["idx_ref"][i], len(A["ref_frames"]) - 1))
        uf = draw_skeleton(A["user_frames"][fu].copy(), A["user_landmarks"][fu])
        rf = draw_skeleton(A["ref_frames"][fr].copy(), A["ref_landmarks"][fr])
        combo = make_side_by_side(uf, rf, per_frame_err[i] >= pause_threshold, float(per_frame_err[i]))
        if combo.shape[:2] != (h, w):
            combo = cv2.resize(combo, (w, h))
        writer.write(combo)
        if per_frame_err[i] >= pause_threshold:
            for _ in range(8):
                writer.write(combo)
    writer.release()

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 2048:
        raise RuntimeError("Видео сравнения не сформировалось или получилось пустым.")
    return out_path

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "user_video_path" not in st.session_state:
    st.session_state.user_video_path = None
if "user_hash" not in st.session_state:
    st.session_state.user_hash = None

tab_analyze, tab_editor = st.tabs(["Анализ", "Добавить/редактировать упражнения"])

with tab_editor:
    st.subheader("Добавление и редактирование упражнений")
    cfg = load_config()
    ids = list(cfg.keys())
    choice = st.selectbox("Упражнение", ["<Новое упражнение>"] + ids)
    is_new = choice == "<Новое упражнение>"

    if is_new:
        element_id = st.text_input("ID (латиница, без пробелов)", value="")
        base = {
            "title": "",
            "reference_video": "",
            "important_joints": {"hip":1.0,"knee":1.0,"ankle":1.0,"torso":1.0,"shoulder":1.0,"elbow":1.0},
            "tips_thresholds_deg": {"minor": 10, "major": 20},
        }
    else:
        element_id = choice
        base = cfg[element_id]

    title = st.text_input("Название", value=base.get("title", ""))
    st.markdown("#### Эталон")
    st.write("Путь:", base.get("reference_video", "(не задан)"))
    ref_upload = st.file_uploader("Загрузить/обновить эталон (mp4/avi/mov/mkv)", type=["mp4","avi","mov","mkv"], key="ref_upload")

    st.markdown("#### Важность зон")
    imp = base.get("important_joints", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        w_hip = st.slider("Таз/бедра", 0.5, 3.0, float(imp.get("hip", 1.0)), 0.1)
        w_knee = st.slider("Колени", 0.5, 3.0, float(imp.get("knee", 1.0)), 0.1)
    with c2:
        w_ankle = st.slider("Лодыжки", 0.5, 3.0, float(imp.get("ankle", 1.0)), 0.1)
        w_torso = st.slider("Корпус", 0.5, 3.0, float(imp.get("torso", 1.0)), 0.1)
    with c3:
        w_sh = st.slider("Плечи", 0.5, 3.0, float(imp.get("shoulder", 1.0)), 0.1)
        w_el = st.slider("Локти", 0.5, 3.0, float(imp.get("elbow", 1.0)), 0.1)

    st.markdown("#### Пороги рекомендаций")
    thr = base.get("tips_thresholds_deg", {"minor": 10, "major": 20})
    t_minor = st.slider("minor", 3, 20, int(thr.get("minor", 10)), 1)
    t_major = st.slider("major", 10, 40, int(thr.get("major", 20)), 1)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Сохранить", type="primary"):
            if not element_id or " " in element_id:
                st.error("ID должен быть без пробелов.")
            else:
                cfg = load_config()
                ref_rel = base.get("reference_video", "")
                if ref_upload is not None:
                    data = ref_upload.read()
                    fname = f"{element_id}_{ref_upload.name}"
                    safe = "".join(ch for ch in fname if ch.isalnum() or ch in "._-")
                    out_path = REF_DIR / safe
                    with open(out_path, "wb") as f:
                        f.write(data)
                    ref_rel = str(out_path.as_posix())
                cfg[element_id] = {
                    "title": title or element_id,
                    "reference_video": ref_rel,
                    "important_joints": {
                        "hip": float(w_hip),
                        "knee": float(w_knee),
                        "ankle": float(w_ankle),
                        "torso": float(w_torso),
                        "shoulder": float(w_sh),
                        "elbow": float(w_el),
                    },
                    "tips_thresholds_deg": {"minor": int(t_minor), "major": int(t_major)},
                }
                save_config(cfg)
                st.success("Сохранено.")
    with colB:
        if (not is_new) and st.button("Удалить"):
            cfg = load_config()
            cfg.pop(element_id, None)
            save_config(cfg)
            st.warning("Удалено.")

with tab_analyze:
    cfg = load_config()
    if not cfg:
        st.warning("Сначала добавьте упражнение во вкладке «Добавить/редактировать упражнения».")
        st.stop()

    ids = list(cfg.keys())
    el = st.selectbox("Упражнение", ids, format_func=lambda k: cfg[k].get("title", k))
    ref_path = cfg[el].get("reference_video", "")
    if not ref_path or not os.path.exists(ref_path):
        st.error(f"Эталон не найден: {ref_path}")
        st.stop()

    user_file = st.file_uploader("Видео пользователя", type=["mp4","avi","mov","mkv"], key="user_upload")
    if user_file is not None:
        data = user_file.read()
        user_hash = md5_bytes(data)
        tmp_dir = Path(tempfile.gettempdir()) / "ai_coach_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        user_path = tmp_dir / f"{user_hash}.mp4"
        if not user_path.exists():
            with open(user_path, "wb") as f:
                f.write(data)
        st.session_state.user_video_path = str(user_path)
        st.session_state.user_hash = user_hash

    if st.button("Анализировать", type="primary"):
        if st.session_state.user_video_path is None:
            st.error("Сначала загрузите видео пользователя.")
            st.stop()
        with st.spinner("Анализ: поза → углы → выравнивание..."):
            st.session_state.analysis = analyze_cached(
                st.session_state.user_hash,
                st.session_state.user_video_path,
                ref_path,
                os.path.getmtime(ref_path),
            )

    if st.session_state.analysis is None:
        st.info("Выберите упражнение, загрузите видео пользователя и нажмите «Анализировать».")
        st.stop()

    A = st.session_state.analysis

    st.subheader("Оценка")
    score = compute_score(A["angle_mae"], A["tempo_err"], cfg[el].get("important_joints", {}))
    st.metric("Баллы", f"{score:.1f} / 100")
    st.caption(f"Рассинхрон темпа: {A['tempo_err']:.2f} с")

    st.subheader("Покадровое сравнение")
    aligned_len = len(A["idx_user"])
    keys = list(A["angle_mae"].keys())
    per_frame_err = np.zeros(aligned_len, dtype=np.float32)
    for i in range(aligned_len):
        s = 0.0
        c = 0
        for k in keys:
            u = A["user_angles"][k][A["idx_user"]][i]
            r = A["ref_angles"][k][A["idx_ref"]][i]
            if np.isfinite(u) and np.isfinite(r):
                s += abs(float(u) - float(r))
                c += 1
        per_frame_err[i] = (s / c) if c else 0.0

    col1, col2 = st.columns([1, 1])
    with col1:
        error_thresh = st.slider("Порог проблемного кадра, °", 0.0, 30.0, 12.0, 0.5)
    with col2:
        show_only_bad = st.checkbox("Только проблемные кадры", value=False)

    candidates = [i for i in range(aligned_len) if (not show_only_bad) or (per_frame_err[i] >= error_thresh)]
    if not candidates:
        st.success("Кадров с ошибкой выше порога не найдено.")
    else:
        j = st.slider("Кадр (по выравниванию DTW)", 0, len(candidates) - 1, 0, 1)
        i = candidates[j]
        fu = max(0, min(A["idx_user"][i], len(A["user_frames"]) - 1))
        fr = max(0, min(A["idx_ref"][i], len(A["ref_frames"]) - 1))
        st.caption(f"Ошибка: {per_frame_err[i]:.1f}° • кадр пользователя: {fu+1} • кадр эталона: {fr+1}")

        uf = draw_skeleton(A["user_frames"][fu].copy(), A["user_landmarks"][fu])
        rf = draw_skeleton(A["ref_frames"][fr].copy(), A["ref_landmarks"][fr])
        combo = make_side_by_side(uf, rf, per_frame_err[i] >= error_thresh, float(per_frame_err[i]))

        import cv2
        st.image(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB), caption="Слева — пользователь, справа — эталон")

    st.markdown("#### Сборка видео сравнения с задержкой на расхождениях")
    if st.button("Собрать видео сравнения"):
        try:
            with st.spinner("Собираю видео..."):
                out_video = build_sync_video_file(A, st.session_state.user_hash, pause_threshold=float(error_thresh))
            if not os.path.exists(out_video) or os.path.getsize(out_video) < 2048:
                st.error("Видео не было сформировано или получилось пустым.")
            else:
                with open(out_video, "rb") as f:
                    video_bytes = f.read()
                st.success("Видео сравнения готово.")
                st.download_button("Скачать видео сравнения", data=video_bytes, file_name="sync_compare.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Не удалось собрать видео: {e}")

    st.subheader("Рекомендации по улучшению")
    tips = generate_ai_recommendations(
        A["angle_mae"],
        cfg[el],
        A["user_angles"],
        A["ref_angles"],
        A["idx_user"],
        A["idx_ref"],
    )

    aligned_user = {k: np.asarray(v)[A["idx_user"]] for k, v in A["user_angles"].items()}
    aligned_ref = {k: np.asarray(v)[A["idx_ref"]] for k, v in A["ref_angles"].items()}
    worst_idx = {}
    for k in aligned_user.keys():
        if k in aligned_ref:
            err = np.abs(aligned_user[k] - aligned_ref[k]).astype(np.float32)
            err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
            worst_idx[k] = int(np.argmax(err))

    if not tips:
        st.write("Рекомендации не сформированы.")
    else:
        for t in tips:
            st.write("• " + t)
            group = group_from_tip(t)
            jkey = representative_joint_key(group) if group else None
            if jkey and jkey in worst_idx:
                i = worst_idx[jkey]
                fu = max(0, min(A["idx_user"][i], len(A["user_frames"]) - 1))
                fr = max(0, min(A["idx_ref"][i], len(A["ref_frames"]) - 1))
                lm_u = A["user_landmarks"][fu]
                lm_r = A["ref_landmarks"][fr]
                frame = draw_joint_overlay(A["user_frames"][fu].copy(), lm_u, lm_r, jkey)
                import cv2
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption=f"Стоп-кадр: {JOINT_NAMES_RU.get(jkey, jkey)} (красная — вы, зелёная — эталон)"
                )
