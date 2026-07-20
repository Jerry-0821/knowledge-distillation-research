"""
student:
Conv -> BN -> ReLU
Conv -> BN -> ReLU -> MaxPool
Conv -> BN -> ReLU
Conv -> BN -> ReLU -> MaxPool
Conv -> BN -> ReLU
AdaptiveAvgPool2d(1)
Flatten
Dropout
Linear


teacher:
Conv -> BN -> ReLU
Conv -> BN -> ReLU -> MaxPool -> Dropout
Conv -> BN -> ReLU
Conv -> BN -> ReLU -> MaxPool -> Dropout
Conv -> BN -> ReLU
Conv -> BN -> ReLU
AdaptiveAvgPool2d(1)
Flatten
Dropout
Linear
"""
from torch import nn

class CIFAR10CNNStudentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # in_channels = RGB channels
            # out_channels = 16 means 16 filters produce 16 feature maps
            # BN must match the number of channels produced by the previous convolution
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(64, 10)
        )
    def forward(self, images):
        return self.network(images)

class CIFAR10CNNTeacherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.20),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.40),
            nn.Linear(128, 10)
        )
    def forward(self, images):
        return self.network(images)
