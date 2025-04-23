# rPPG Benchmarking & Real-Time Tester

Проект предоставляет инфраструктуру для оценки качества и реального применения алгоритмов rPPG (remote photoplethysmography) — технологии извлечения частоты сердечных сокращений (ЧСС) по видео лица.

## 📄 Структура проекта

```
rppg_project/
├── rppg_benchmark/              # Библиотека для бенчмаркинга
│   ├── interfaces.py            # Интерфейс моделей rPPG
│   ├── metrics.py               # Метрики: MAE, RMSE, SNR, Pearson r
│   ├── datasets.py              # Загрузка изображений/видео + HR
│   ├── benchmark.py             # Прогон моделей по датасетам
│   └── adapters/
│       └── yarppg_adapter.py    # Обёртка для библиотеки yarppg
│
├── rppg_tester.py               # Онлайн тестер ЧСС через веб-камеру
└── README.md                    # Документация проекта
```

## 🔹 Назначение

- **rppg_benchmark** — модуль оценки моделей по датасетам (с эталонным HR или ppg сигналом).
- **rppg_tester.py** — консольное приложение, использующее модель для онлайн измерения HR в реальном времени через веб-камеру.

## 👁️ Интерфейс моделей: `IRPPGModel`

Любая модель rPPG должна реализовать следующий интерфейс (см. `interfaces.py`):

```python
class IRPPGModel:
    def reset(self): ...
    def process_frame(self, frame_rgb): ...
    def get_ppg(self): ...
    def get_hr(self, fps): ...  # по умолчанию использует FFT
```

## 🌐 Пример использования: офлайн

```python
from rppg_benchmark.adapters.yarppg_adapter import YarppgAdapter
from rppg_benchmark.datasets import FrameDataset
from rppg_benchmark.benchmark import RPPGBenchmark

model = YarppgAdapter()
dataset = FrameDataset("dataset_frames/")
bench = RPPGBenchmark(dataset, fps=30)

report = bench.evaluate(model)
print(report)
```

## ⏱️ Пример использования: онлайн

```bash
python rppg_tester.py                             # использует YarppgAdapter
python rppg_tester.py my_lib.adapters:MyAdapter   # кастомная модель
```

## ⚖️ Метрики качества (файл `metrics.py`)

| Метрика     | Описание                              |
|-------------|----------------------------------------|
| MAE         | Средняя абсолютная ошибка (в BPM)      |
| RMSE        | Корень средней квадратичной ошибки     |
| Pearson r   | Корреляция между эталоном и моделью    |
| SNR         | Отношение сигнал/шум в dB              |

## 📚 Формат датасета

Файлы изображений вида:
```
img_001_hr_73.4.png
```
Из имени извлекается значение HR.

## ✅ Преимущества

- ✅ Единый API для всех моделей
- ✅ Гибкость: можно подставлять свои модели без изменения ядра
- ✅ Возможность офлайн-бенчмарка и онлайн-проверки

## 📆 Зависимости

- Python >= 3.9
- OpenCV
- NumPy
- yarppg (если используется YarppgAdapter)

## 🔍 TODO / планы

- [ ] Поддержка видеофайлов в `datasets.py`
- [ ] Метрики на уровне ppg-сигнала
- [ ] Поддержка CI и unit-тестов моделей
- [ ] HTML-отчёты для сравнения моделей

---

✌️ Проект разработан для исследовательских и прикладных целей анализа rPPG. Добро пожаловать в соавторы!


