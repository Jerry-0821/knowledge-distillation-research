#checks model output shape only

import torch

from kd_research.models import (
    CIFAR10CNNStudentModel,
    CIFAR10CNNTeacherModel,
)


def test_cifar10_student_outputs_ten_logits() -> None:
    model = CIFAR10CNNStudentModel()
    images = torch.randn(size=(4, 3, 32, 32))

    logits = model(images)
    # 4 images(batch size) and 10 classes
    assert logits.shape == (4, 10)

def test_cifar10_teacher_outputs_ten_logits() -> None:
    model = CIFAR10CNNTeacherModel()
    images = torch.randn(4, 3, 32, 32)

    logits = model(images)
    assert logits.shape == (4, 10)