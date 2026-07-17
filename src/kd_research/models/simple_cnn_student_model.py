from torch import nn

class SimpleCNNStudentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, 10),
        )

    def forward(self, images):
        return self.network(images)
