import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x

if __name__ == '__main__':
    model = SRCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    