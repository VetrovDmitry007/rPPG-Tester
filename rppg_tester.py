#!/usr/bin/env python3
"""
rppg_tester.py
==============

Консольный «полевой» скрипт-проверка ЧСС в реальном времени,
построенный на едином интерфейсе IRPPGModel из пакета
`rppg_benchmark`.

• По умолчанию использует адаптер YarppgAdapter, но можно
  подставить любой другой класс, совместимый с IRPPGModel:
      python rppg_tester.py path.to.MyAdapter

• Файл settings.json хранит:
    - disable_app        : включить / выключить приложение
    - check_in_interval  : пауза между измерениями (мин)
    - is_athlete         : пониженный порог HR

Для работы нужен установленный rppg_benchmark и OpenCV.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import List, Tuple, Type

import cv2
import numpy as np

from rppg_benchmark.interfaces import IRPPGModel

# ────────────────────────────────
# Настройки и константы
# ────────────────────────────────
SETTINGS_FILE = "settings.json"
MEASURE_SECONDS = 100

DEFAULT_MODEL_PATH = "rppg_benchmark.adapters.yarppg_adapter:YarppgAdapter"

MINDFULNESS_EXERCISES = [
    ("Deep Breathing",
     "Inhale 4 s → hold 4 s → exhale 6 s. Repeat 5 cycles."),
    ("Body Scan",
     "Close your eyes and slowly bring attention from toes to head."),
    ("5-4-3-2-1 Grounding",
     "5 things you see, 4 feel, 3 hear, 2 smell, 1 taste."),
    ("Gratitude Reflection",
     "Spend 2 min thinking of three things you are grateful for."),
    ("Mindful Observation",
     "Focus 1 min on a nearby object – colour, texture, shape."),
]


# ────────────────────────────────
# Вспомогательные классы / функции
# ────────────────────────────────
class FpsTracker:
    """Скользящий счётчик FPS (~1 с. окно)."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._frames = 0
        self.fps: float = 0.0

    def tick(self) -> None:
        self._frames += 1
        dt = time.time() - self._t0
        if dt >= 1.0:
            self.fps = self._frames / dt
            self._frames = 0
            self._t0 = time.time()


def load_settings() -> Tuple[bool, int, bool]:
    """Читает или создаёт JSON-файл настроек."""
    defaults = {
        "disable_app": False,
        "check_in_interval": 60,   # минут
        "is_athlete": False,
    }
    path = Path(SETTINGS_FILE)
    if path.exists():
        cfg = json.loads(path.read_text("utf-8"))
    else:
        cfg = defaults.copy()
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    return (
        bool(cfg.get("disable_app", False)),
        int(cfg.get("check_in_interval", 60)),
        bool(cfg.get("is_athlete", False)),
    )


def is_hr_anomalous(hr: float, athlete: bool = False) -> Tuple[bool, str]:
    """Сравнивает HR с порогом и формирует отчёт."""
    th = 80.0 if athlete else 100.0
    return hr > th, f"HR={hr:.1f} BPM (threshold {th} BPM)"


def dynamic_import(path: str) -> Type[IRPPGModel]:
    """
    Загружает класс по строке формата ``module.sub:ClassName``.
    Поднимает исключение, если импорт не удался или класс
    не наследует IRPPGModel.
    """
    if ":" not in path:
        raise ValueError("model path must be 'module.sub:ClassName'")
    mod_name, cls_name = path.split(":", 1)
    module: ModuleType = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    if not issubclass(cls, IRPPGModel):
        raise TypeError(f"{cls} does not implement IRPPGModel")
    return cls


# ────────────────────────────────
# Основная логика измерения
# ────────────────────────────────
def check_in(model: IRPPGModel, *, athlete: bool = False) -> None:
    """Одна 20-секундная сессия измерения HR."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open camera")
        return

    model.reset()
    tracker = FpsTracker()
    hrs: List[float] = []

    print(f"[*] Measuring HR for {MEASURE_SECONDS} s …")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    t_start = time.time()
    while time.time() - t_start < MEASURE_SECONDS:
        ok, frame_bgr = cam.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model.process_frame(frame_rgb)
        tracker.tick()

        if tracker.fps > 0:
            hr_bpm = model.get_hr(tracker.fps)
            # print(f"HR={hr_bpm:.1f} BPM ({tracker.fps:.1f} FPS)")
            if np.isfinite(hr_bpm) and hr_bpm > 0:
                hrs.append(hr_bpm)

    cam.release()

    if not hrs:
        print("[WARN] No valid HR values collected")
        return

    avg_hr = float(np.mean(hrs))
    bad, report = is_hr_anomalous(avg_hr, athlete=athlete)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if bad:
        exercise = random.choice(MINDFULNESS_EXERCISES)
        print(f"[{timestamp}] 🚨 High HR! {report}")
        print(f"    → Try: {exercise[0]} — {exercise[1]}")
    else:
        print(f"[{timestamp}] ✅ HR normal. {report}")


# ────────────────────────────────
# Точка входа
# ────────────────────────────────
def main(model_path: str = DEFAULT_MODEL_PATH) -> None:
    """Запускает циклический мониторинг HR."""
    ModelCls = dynamic_import(model_path)

    disabled, interval_min, athlete = load_settings()
    if disabled:
        print("[*] App disabled in settings.json. Exit.")
        sys.exit(0)

    model = ModelCls()
    print(f"[*] Check-in every {interval_min} min  (athlete={athlete})")

    try:
        while True:
            check_in(model, athlete=athlete)
            time.sleep(interval_min * 60)
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user (Ctrl+C)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    main(path)
