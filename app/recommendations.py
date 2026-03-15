import numpy as np

GROUP_KEYS = {
    "hip": ["hip_left", "hip_right"],
    "knee": ["knee_left", "knee_right"],
    "ankle": ["ankle_left", "ankle_right"],
    "torso": ["torso"],
    "shoulder": ["shoulder_left", "shoulder_right"],
    "elbow": ["elbow_left", "elbow_right"],
}
GROUP_NAMES = {
    "hip": "таз/бедра",
    "knee": "колени",
    "ankle": "лодыжки",
    "torso": "корпус",
    "shoulder": "плечи",
    "elbow": "локти",
}
PHASE_NAMES = {"start": "начальная фаза", "mid": "средняя фаза", "end": "финальная фаза"}

def _phase_label(pos: float) -> str:
    if pos < 0.33:
        return "start"
    if pos < 0.66:
        return "mid"
    return "end"

def _avg(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return np.nan
    return float(np.mean(values))

def summarize_group_errors(angle_mae: dict, important_joints: dict):
    stats = []
    for group, keys in GROUP_KEYS.items():
        err = _avg([angle_mae.get(k, np.nan) for k in keys])
        if np.isfinite(err):
            w = float(important_joints.get(group, 1.0))
            stats.append({"group": group, "name": GROUP_NAMES[group], "err": err, "weight": w, "priority": err * w})
    stats.sort(key=lambda x: x["priority"], reverse=True)
    return stats

def worst_phase_by_group(user_angles: dict, ref_angles: dict, idx_user: list, idx_ref: list):
    result = {}
    if not idx_user or not idx_ref:
        return result
    T = len(idx_user)
    for group, keys in GROUP_KEYS.items():
        per_frame = []
        for i in range(T):
            vals = []
            for k in keys:
                if k not in user_angles or k not in ref_angles:
                    continue
                u = user_angles[k][idx_user][i]
                r = ref_angles[k][idx_ref][i]
                if np.isfinite(u) and np.isfinite(r):
                    vals.append(abs(float(u) - float(r)))
            per_frame.append(float(np.mean(vals)) if vals else np.nan)
        arr = np.asarray(per_frame, dtype=np.float32)
        if np.isfinite(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
            worst_i = int(np.argmax(arr))
            pos = worst_i / max(1, T - 1)
            result[group] = {"index": worst_i, "phase": _phase_label(pos), "phase_ru": PHASE_NAMES[_phase_label(pos)], "err": float(arr[worst_i])}
    return result

def _general_recs(group: str, err: float, phase_ru: str, title: str):
    recs = []
    if group == "knee":
        recs.append(f"Следите за одинаковой глубиной сгибания коленей в фазе: **{phase_ru}**.")
        recs.append("Избегайте резкого распрямления коленей сразу после нижней точки.")
        if err > 18:
            recs.append("Потренируйте медленные повторения с паузой в нижней точке, чтобы зафиксировать правильный угол коленей.")
            recs.append("Проверьте, не уводите ли колени внутрь или наружу относительно траектории эталона.")
    elif group == "hip":
        recs.append(f"Контролируйте посадку таза и работу бедер в фазе: **{phase_ru}**.")
        recs.append("Сохраняйте устойчивое положение таза, не проваливайтесь слишком рано вниз.")
        if err > 18:
            recs.append("Попробуйте выполнять упражнение у зеркала, чтобы отслеживать глубину ухода таза.")
            recs.append("Разбейте движение на фазы: отдельно отработайте вход в нижнюю позицию и выход из неё.")
    elif group == "torso":
        recs.append(f"Держите корпус стабильнее в фазе: **{phase_ru}**.")
        recs.append("Не заваливайтесь вперёд и не переразгибайтесь назад относительно эталона.")
        if err > 18:
            recs.append("Добавьте медленные повторения с акцентом на нейтральное положение спины.")
            recs.append("Снимайте упражнение сбоку, чтобы контролировать угол наклона корпуса.")
    elif group == "ankle":
        recs.append(f"Проверьте работу стопы и лодыжек в фазе: **{phase_ru}**.")
        recs.append("Следите за распределением веса по стопе и устойчивой опорой.")
        if err > 18:
            recs.append("Попробуйте выполнять движение в более медленном темпе, контролируя пятку и носок.")
            recs.append("Проверьте, не происходит ли преждевременный отрыв пятки от опоры.")
    elif group == "shoulder":
        recs.append(f"Соберите плечевой пояс и стабилизируйте плечи в фазе: **{phase_ru}**.")
        recs.append("Избегайте лишнего подъёма плеч или раскачивания плечевого пояса.")
        if err > 18:
            recs.append("Сделайте несколько повторов без ускорения, концентрируясь только на линии плеч.")
            recs.append("Сопоставьте положение плеч в начальной и финальной фазе с эталоном.")
    elif group == "elbow":
        recs.append(f"Проконтролируйте работу локтей в фазе: **{phase_ru}**.")
        recs.append("Старайтесь не сгибать и не разгибать руки раньше нужного момента.")
        if err > 18:
            recs.append("Отработайте движение рук отдельно от нижней части тела, затем объедините фазы.")
            recs.append("Следите, чтобы локти не уходили в сторону относительно эталонной траектории.")
    return recs

def generate_ai_recommendations(angle_mae: dict, cfg_el: dict, user_angles: dict, ref_angles: dict, idx_user: list, idx_ref: list):
    if not angle_mae:
        return []
    title = cfg_el.get("title", "упражнение")
    thresholds = cfg_el.get("tips_thresholds_deg", {"minor": 10, "major": 20})
    minor = float(thresholds.get("minor", 10))
    major = float(thresholds.get("major", 20))
    important = cfg_el.get("important_joints", {})

    stats = summarize_group_errors(angle_mae, important)
    phases = worst_phase_by_group(user_angles, ref_angles, idx_user, idx_ref)

    tips = []
    added = set()
    for item in stats:
        group = item["group"]
        err = item["err"]
        phase_ru = phases.get(group, {}).get("phase_ru", "ключевая фаза")
        if err < minor:
            continue

        if group == "knee":
            if err >= major:
                tips.append(f"В упражнении «{title}» заметная ошибка в зоне **коленей** (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**.")
            else:
                tips.append(f"Немного подправьте **угол коленей** в фазе: **{phase_ru}**. Текущее отклонение ≈ {err:.1f}°.")
        elif group == "hip":
            if err >= major:
                tips.append(f"Ключевая неточность — **таз/бедра** (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**.")
            else:
                tips.append(f"Чуть скорректируйте **положение таза/бедер** в фазе: **{phase_ru}**. Отклонение ≈ {err:.1f}°.")
        elif group == "torso":
            if err >= major:
                tips.append(f"**Корпус** отклоняется от эталона (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**.")
            else:
                tips.append(f"Сделайте **корпус** более стабильным в фазе: **{phase_ru}**. Отклонение ≈ {err:.1f}°.")
        elif group == "ankle":
            tips.append(f"Есть отличие в работе **лодыжек/стопы** (≈ {err:.1f}°), сильнее в фазе: **{phase_ru}**.")
        elif group == "shoulder":
            tips.append(f"Зона **плеч** отличается от эталона (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**.")
        elif group == "elbow":
            tips.append(f"Есть ошибка в работе **локтей** (≈ {err:.1f}°), сильнее в фазе: **{phase_ru}**.")

        for rec in _general_recs(group, err, phase_ru, title):
            if rec not in added:
                tips.append(rec)
                added.add(rec)

    if not tips:
        tips.append("Техника близка к эталону. Основной резерв — улучшить синхронность и чистоту переходов между фазами.")
        tips.append("Попробуйте снимать больше дублей с одинакового ракурса, чтобы точнее сравнивать движение с эталоном.")
    return tips[:10]
