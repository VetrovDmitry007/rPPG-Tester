"""
Адаптер для модели PhysFormer/ViT_ST_ST_Compact3_TDC_gra_sharp
https://github.com/ZitongYu/PhysFormer
"""

from rppg_benchmark.interfaces import IRPPGModel
import torch
import numpy as np
from PhysFormer.model import ViT_ST_ST_Compact3_TDC_gra_sharp


class PhysFormerAdapter(IRPPGModel):
    def __init__(self):
        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
        ckpt_path = "models/PURE_PhysFormer_DiffNormalized.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.frames = []

    def reset(self):
        self.frames.clear()

    def process_frame(self, frame_rgb, ts=None):
        self.frames.append(frame_rgb)

    def get_ppg(self):
        if len(self.frames) < 160:
            return np.array([], dtype=np.float32)
        x = self._preprocess(self.frames)
        with torch.no_grad():
            rppg, *_ = self.model(x, gra_sharp=None)
        return rppg.squeeze().cpu().numpy()

    def _preprocess(self, frames):
        arr = np.stack(frames, axis=0)  # T,H,W,3
        arr = arr.transpose(3, 0, 1, 2)[None]  # 1,3,T,H,W
        arr = arr.astype(np.float32) / 255.0
        return torch.tensor(arr)

    def get_hr(self, fps: float):
        pass

if __name__ == "__main__":
    adapter = PhysFormerAdapter()
    print(adapter.model)

