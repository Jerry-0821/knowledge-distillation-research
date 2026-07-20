"""Model definitions for the knowledge distillation research project."""

from kd_research.models.cifar_cnn_models import (
    CIFAR10CNNStudentModel, CIFAR10CNNTeacherModel,
)
from kd_research.models.simple_cnn_student_model import SimpleCNNStudentModel
from kd_research.models.simple_cnn_teacher_model import SimpleCNNTeacherModel

__all__ = [
    "CIFAR10CNNStudentModel",
    "CIFAR10CNNTeacherModel",
    "SimpleCNNStudentModel",
    "SimpleCNNTeacherModel",
]