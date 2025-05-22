from emo_state.metrics import *
import matplotlib.pyplot as plt

def show_metrics(pred_ppg, ref_ppg, fps):
    # Вычислить стандартные метрики ошибки и сходства
    print("MAE       =", mae(pred_ppg, ref_ppg))
    print("MAPE       =", mape(pred_ppg, ref_ppg))
    print("SMAPE       =", smape(pred_ppg, ref_ppg))
    print("RMSE      =", rmse(pred_ppg, ref_ppg))
    print("Pearson r =", corr(pred_ppg, ref_ppg))
    print("SNR (дБ)  =", snr(pred_ppg, ref_ppg))
    print(f"\nСредний (ЧСС) HR по пикам (BPM): ref = {ref_mean_hr(ref_ppg, fps=fps)}, pred = {pred_mean_hr(pred_ppg, fps=fps)} ")
    print(f'Ошибка среднего (ЧСС) HR (BPM): {errors_mean_hr(ref_ppg, pred_ppg, fps=fps)}%')
    print(f"Медианный (ЧСС) HR (BPM): ref = {ref_median_hr(ref_ppg, fps=fps)}, pred = {pred_median_hr(pred_ppg, fps=fps)} ")
    print(f'Ошибка медианного (ЧСС) HR (BPM): {errors_median_hr(ref_ppg, pred_ppg, fps=fps)}%')
    print(f'Вариабельность сердечного ритма HRV: ref = {ref_hrv(ref_ppg, fps=fps)}, pred = {pred_hrv(pred_ppg, fps=fps)}')

    # Визуальная оценка
    # plt.figure(figsize=(10,4))
    # plt.plot(ref_ppg,  label="Эталонный PPG")
    # plt.plot(pred_ppg, label="Предсказанный PPG")
    # plt.legend()
    # plt.xlabel("Отсчёт")
    # plt.ylabel("Амплитуда")
    # plt.title("Сравнение эталонного и предсказанного PPG")
    # plt.tight_layout()
    # plt.show()