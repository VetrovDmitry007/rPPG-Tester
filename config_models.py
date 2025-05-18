
# Пути к видео и csv-файлам
# *************************
path_csv = "data/SCAMPS_smail/output_data/ppg_1.csv"
video_path = "data/SCAMPS_smail/output_data/video_1.avi"

# path_csv = "data/sample/sample_vitals_2.csv"
# video_path = "data/sample/sample_video_2.mp4"

# path_csv = "data/UBFC-Phys/bvp_s11_T1.csv"
# video_path = "data/UBFC-Phys/vid_s11_T1.avi"

# *************************


# Пути к моделям
# *************************
# supervised_methods
# MODEL_PATH = "rppg_benchmark.adapters.phys_former_adapter:PhysFormerAdapter"
# MODEL_PATH = "rppg_benchmark.adapters.deep_phys_adapter:DeepPhysAdapter"
# MODEL_PATH = "rppg_benchmark.adapters.tscan_adapter:TSCANAdapter"

# unsupervised_methods
# MODEL_PATH = "rppg_benchmark.adapters.pos_adapter:PosAdapter"
MODEL_PATH = "rppg_benchmark.adapters.chrome_adapter:ChromeAdapter"

# ************************