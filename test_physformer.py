"""
Пример использования модели PhysFormer/ViT_ST_ST_Compact3_TDC_gra_sharp

Результат работы модели -- предсказанный временной ряд, отражающим фотоплетизмографическую волну,
         извлечённую моделью PhysFormer из видеопоследовательности
"""
import cv2
from rppg_tester import dynamic_import

MODEL_PATH = "rppg_benchmark.adapters.phys_former_adapter:PhysFormerAdapter"
ModelCls = dynamic_import(MODEL_PATH)
model = ModelCls()

# print(model.model)

model.reset()

cap = cv2.VideoCapture('data/SCAMPS_smail/output_data/video.avi')  # веб-камера
for _ in range(160):       # требуется 160 кадров
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.process_frame(frame_rgb, fps=30)

cap.release()

ppg = model.get_ppg()
# print(ppg.shape, ppg[:5])  # например (160,), [0.13, 0.09, ...]
print(ppg.shape, ppg)  # например (160,), [0.13, 0.09, ...]