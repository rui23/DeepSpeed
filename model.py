import torch
import numpy as np

class FashionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Linear( 784, 300),
            torch.nn.ReLU(),
            torch.nn.Linear( 300, 10),
        )

    def forward(self, batch_x):
        return self.seq(batch_x)
    
def img_transform(img): #图片预处理
    img = np.asarray(img) / 255
    return torch.tensor(img, dtype=torch.float32).view(-1)