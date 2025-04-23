#!/usr/bin/env python3
"""
PulsePause Console Version

Этот скрипт периодически измеряет частоту сердечных сокращений (HR)
с помощью веб-камеры и алгоритма rPPG (Remote Photoplethysmography).
Вместо GUI все взаимодействие происходит через консоль.
"""
import os
import json
import time
import random
import sys
import warnings

import numpy as np
import yarppg
import cv2

# Путь к файлу с настройками (JSON)
SETTINGS_FILE = "settings.json"

# Список упражнений для релаксации при высокой ЧСС
MINDFULNESS_EXERCISES = [
    {
        "name": "Deep Breathing",
        "description": (
            "Take a deep breath in for 4 seconds, hold it for 4 seconds, "
            "then exhale for 6 seconds. Repeat 5 cycles."
        ),
    },
    {
        "name": "Body Scan",
        "description": (
            "Close your eyes and slowly bring attention to different parts "
            "of your body, from your toes to your head."
        ),
    },
    {
        "name": "5-4-3-2-1 Grounding",
        "description": (
            "Identify 5 things you can see, 4 things you can feel, 3 things "
            "you can hear, 2 things you can smell, and 1 thing you can taste."
        ),
    },
    {
        "name": "Gratitude Reflection",
        "description": (
            "Spend 2 minutes reflecting on 3 things you’re grateful for and why."
        ),
    },
    {
        "name": "Mindful Observation",
        "description": (
            "Choose a nearby object and focus on it for 1 minute: "
            "note its color, texture, shape and small details."
        ),
    },
]

# Инициализация rPPG-модуля
rppg = yarppg.Rppg()


def load_settings():
    """
    Читает файл настроек JSON (если существует) и возвращает три параметра:
      - disable_app: bool (отключить приложение)
      - check_in_interval: int (интервал между измерениями в минутах)
      - is_athlete: bool (специальный порог для спортсменов)
    Если файла нет, создаёт его с дефолтными значениями.
    """
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            cfg = json.load(f)
    else:
        # Дефолтные настройки при первом запуске
        cfg = {"disable_app": False, "check_in_interval": 60, "is_athlete": False}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(cfg, f, indent=2)

    # Извлекаем параметры с защитой от отсутствующих ключей
    disabled = cfg.get("disable_app", False)
    interval = cfg.get("check_in_interval", 60)
    athlete = cfg.get("is_athlete", False)
    return disabled, interval, athlete


def is_heart_rate_anomalous(hr: float, age_group: str = "adult"):
    """
    Проверяет, превышает ли средняя частота пульса (hr)
    пороговое значение для заданной группы:
      - adult   → 100 BPM
      - athlete → 80 BPM

    Возвращает кортеж (bool, str):
      - True/False – факт аномалии
      - Сообщение с измеренным HR и порогом
    """
    # Определяем пороги для разных групп
    thresholds = {"adult": 100, "athlete": 80}
    th = thresholds.get(age_group, thresholds["adult"])
    # Флаг аномалии и текстовый отчёт
    return hr > th, f"HR={hr:.1f} BPM (threshold {th} BPM)"


def check_in(is_athlete: bool = False):
    """
    Выполняет измерение HR в течение 20 секунд.
    - Подавляет специфичные предупреждения numpy/yarppg
    - Считывает кадры с камеры
    - С помощью алгоритма rPPG вычисляет мгновенную частоту
    - Собирает только валидные значения в список
    - По завершению выводит средний HR и
      либо совет по упражнению, либо сообщение о норме.
    """
    # Открываем веб-камеру (ID 0)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Не удалось открыть камеру")
        return

    tracker = yarppg.FpsTracker()  # Трекер FPS для пересчёта HR
    start_time = time.time()
    hrs = []  # список валидных частот

    print("[*] Начало замера HR (20 сек)...")

    # Подавляем шумные предупреждения, связанные с пустыми массивами
    warnings.filterwarnings("ignore", message="Mean of empty slice.")
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

    # Захват кадров в течение 20 секунд
    while time.time() - start_time < 20:
        ret, frame_bgr = cam.read()
        if not ret:
            break
        # Конвертируем BGR → RGB для rPPG
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = rppg.process_frame(frame_rgb)
        tracker.tick()

        # Пересчёт мгновенного HR в BPM
        if res.hr > 0:
            hr_bpm = 60 * tracker.fps / res.hr
            # Добавляем только конечные числа (float, не inf/NaN)
            if np.isfinite(hr_bpm):
                hrs.append(hr_bpm)

    cam.release()

    # Если за 20 сек не набралось ни одного измерения
    if not hrs:
        print("[WARN] Не удалось измерить ни одного показания HR")
        return

    # Вычисляем среднюю частоту
    avg_hr = sum(hrs) / len(hrs)
    age_group = "athlete" if is_athlete else "adult"
    anomalous, report = is_heart_rate_anomalous(avg_hr, age_group)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Выводим результат в консоль
    if anomalous:
        # Выбираем случайное упражнение из списка
        exercise = random.choice(MINDFULNESS_EXERCISES)
        print(f"[{timestamp}] 🚨 High HR detected! {report}")
        print(
            f"    → Suggested: {exercise['name']} — {exercise['description']}"
        )
    else:
        print(f"[{timestamp}] ✅ HR normal. {report}")


def main():
    """
    Точка входа в приложение:
     - Загружает настройки
     - Если приложение отключено, завершается
     - Иначе в бесконечном цикле раз в заданный интервал
       вызывает функцию check_in и ждёт следующего запуска
    """
    disabled, interval_min, is_athlete = load_settings()
    if disabled:
        print("[*] Приложение отключено в настройках. Выход.")
        sys.exit(0)

    print(f"[*] Запуск check-in каждые {interval_min} мин. (athlete={is_athlete})")
    try:
        while True:
            check_in(is_athlete=is_athlete)
            time.sleep(interval_min * 60)
    except KeyboardInterrupt:
        # Обработка Ctrl+C
        print("\n[EXIT] Завершение по Ctrl+C.")


if __name__ == "__main__":
    main()
