from torch import nn


class SimpleCNNTeacherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 10),
        )

    def forward(self, images):
        return self.network(images)
    
#The teacher channel numbers are chosen as a simple larger version of the student: 
# student uses 8 channels, teacher uses 16 then 32 channels. 
# This does not guarantee better performance, so we need to train and test the teacher.