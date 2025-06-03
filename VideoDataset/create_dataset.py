"""
1. Возвращает список видеофайлов в директории
2. Разделить фреймы видео на кадры по 60 секунд
3. Преобразование набора видео в HRV-вектор с сохранением в CSV файл
"""
import os
from pathlib import Path
import pandas as pd

from rppg_benchmark.datasets import load_dataset
from rppg_benchmark.rppg_analyzer import RPPGSignalAnalyzer
from rppg_tester import dynamic_import
from config_models import MODEL_PATH, video_path, path_csv

# Загрузка модели
ModelCls = dynamic_import(MODEL_PATH)
model = ModelCls()


def video_to_hrv(video_path):
    """
    Преобразование видео в HRV с записью в CSV

    :param video_path: путь к видео -- '../data/UBFC-Phys/vid_s7_T2.avi'
    """
    stem = video_path.stem
    print(f'Обработка видео: {video_path}')
    video_frames = load_dataset(video_path)
    # video_frames = load_dataset("../data/sample/sample_video_1.mp4")
    fps = int(video_frames.fps)

    # Разделить фреймы видео на кадры по 60 секунд
    video_frames_60_sec = [ list(video_frames)[i:i+fps*60] for i in range(0, len(video_frames), fps*60) ]

    ls_metrics = []
    for cn, dataset in enumerate(video_frames_60_sec, start=1):
        dc_result = {'file': stem, 'frame': cn}
        print(f'Обрабатывается фрейм {cn}/{len(video_frames_60_sec)}')
        model.reset()
        model.load_dataset(dataset)
        pred_ppg = model.get_ppg()
        analyzer = RPPGSignalAnalyzer(pred_ppg, fps=video_frames.fps, threshold=20)
        if analyzer.flag:
            dc_result.update(analyzer.index_hrv)
            print(f'{analyzer.index_hrv=}')
            ls_metrics.append(dc_result)

    df = pd.DataFrame(ls_metrics)

    # Проверяем наличие файла и сохраняем с нужными параметрами
    csv_path = 'dataset_to_autoencoder.csv'
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)


def iterable_video_file(video_folder):
    """
    Возвращает список видеофайлов в директории
    """
    video_dir = Path(video_folder)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions and f.is_file()]
    return video_files


def start_video_to_hrv(video_folder):
    for video_path in iterable_video_file(video_folder):
        video_to_hrv(video_path)


if __name__ == '__main__':
    start_video_to_hrv('../data/UBFC-Phys/')


