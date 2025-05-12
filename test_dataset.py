"""
Пример использования бенчмарка
"""

from rppg_benchmark.datasets import load_dataset
from rppg_benchmark.benchmark import RPPGBenchmark
from rppg_tester import dynamic_import

DEFAULT_MODEL_PATH = "rppg_benchmark.adapters.yarppg_adapter:YarppgAdapter"
ModelCls = dynamic_import(DEFAULT_MODEL_PATH)
model = ModelCls()

# для видео MPG с PPG-CSV
ds = load_dataset("data/SCAMPS_smail/output_data/video.avi", vitals_csv="data/SCAMPS_smail/output_data/ppg.csv")
bench = RPPGBenchmark(ds, fps=30.0)

report = bench.evaluate(model)
print(report)