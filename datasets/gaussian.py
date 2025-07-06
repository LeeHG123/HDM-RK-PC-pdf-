import torch
import math


class GaussianDataset(torch.utils.data.Dataset):
    """y = φ(x) + ε  (φ: standard normal pdf)"""

    def __init__(self, num_data, num_points, seed: int = 42):
        super().__init__()
        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        
        # 1. x 좌표 그리드를 생성합니다. (-10에서 10까지)
        self.x = (
            torch.linspace(-10., 10., steps=num_points)
            .unsqueeze(0)
            .repeat(num_data, 1)
        )

        # 2. x 좌표를 사용하여 표준 정규 분포의 PDF(phi) 값을 계산합니다.
        #    이것이 함수의 기본 형태(bell curve)가 됩니다.
        phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * self.x ** 2)

        # 3. 각 함수에 더해줄 노이즈(eps)를 생성합니다.
        eps = torch.normal(mean=0., std=0.01, size=(self.x.shape[0], 1)).repeat(1, self.num_points) * 0.1

        # 4. 최종 데이터셋은 PDF 값에 노이즈를 더한 형태가 됩니다.
        self.dataset = phi + eps

    # --------------------------------------------------
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        PDF_MAX = 2.0 / math.sqrt(2 * math.pi)     # ≈ 0.39894228
        return (
            self.x[idx].unsqueeze(-1),             # (N,1)
            (self.dataset[idx] / PDF_MAX).unsqueeze(-1)   # (N,1)
        )