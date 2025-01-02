from typing import NamedTuple, Literal, TypeAlias

import numpy as np


WeightsInitializerName: TypeAlias = Literal[
    "random", "uniform", "xavier_uniform", "xavier_normal", "he_uniform", "he_normal"
]

OptimizerName: TypeAlias = Literal[
    "gd",
    "gradient_descent",
    "nag",
    "nmgd",
    "nesterov_momentum_gradient_descent",
    "rmsprop",
    "adam",
]

ActivationName: TypeAlias = Literal["linear", "relu", "sigmoid", "softmax"]

LayerTypeName: TypeAlias = Literal["input", "dropout", "dense"]

LossName: TypeAlias = Literal["binarycrossentropy", "crossentropy", "mse"]

ALL_OPTIMIZERS: list[OptimizerName] = [
    "gd",
    "gradient_descent",
    "nag",
    "nmgd",
    "nesterov_momentum_gradient_descent",
    "rmsprop",
    "adam",
]

ALL_ACTIVATIONS: list[ActivationName] = ["linear", "relu", "sigmoid", "softmax"]

ALL_LOSSES: list[LossName] = ["binarycrossentropy", "crossentropy", "mse"]

ALL_LAYER_TYPES: list[LayerTypeName] = ["input", "dropout", "dense"]

Metrics: TypeAlias = tuple[float, float, float, float]

MetricNames: TypeAlias = Literal['accuracy', 'precision', 'recall', 'f1_score']

class History:
    __list_metrics: tuple[MetricNames, ...] = ('accuracy', 'precision', 'recall', 'f1_score')
    def __init__(self) -> None:
        self.__loss: list[float] = []
        self.__val_loss: list[float] = []
        self.__metrics: np.ndarray = np.empty((0, 4))
        self.__val_metrics: np.ndarray = np.empty((0, 4)) 

    def __get_selected_metrics(self, metrics: np.ndarray, *list_: MetricNames) -> np.ndarray:
        if not list_:
            list_ = self.__list_metrics
        list_idx = [ self.__list_metrics.index(m) for m in list_ ]
        selected_ = metrics[:, list_idx]
        return selected_.T if len(list_) > 1 else selected_

    def add_loss(self, loss: float) -> None:
        self.__loss.append(loss)

    def add_val_loss(self, loss: float) -> None:
        self.__val_loss.append(loss)

    def add_metrics(self, metrics: Metrics) -> None:
        self.__metrics = np.append(self.__metrics, [metrics], axis=0)

    def add_val_metrics(self, metrics: Metrics) -> None:
        self.__val_metrics = np.append(self.__val_metrics, [metrics], axis=0)

    @property
    def loss(self) -> list[float]:
        return self.__loss

    @property
    def val_loss(self) -> list[float]:
        return self.__val_loss

    def metrics(self, *selected_metrics: MetricNames) -> np.ndarray:
        return self.__get_selected_metrics(self.__metrics, *selected_metrics)

    def val_metrics(self, *selected_metrics: MetricNames) -> np.ndarray:
        return self.__get_selected_metrics(self.__val_metrics, *selected_metrics)


class DropoutLayerData(NamedTuple):
    ratio: float
    output_dim: int
    type: LayerTypeName


class DenseLayerData(NamedTuple):
    weights: np.ndarray
    bias: np.ndarray
    activation: ActivationName
    output_dim: int
    type: LayerTypeName


def layer_standardization(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + 1e-7)


def batch_standardization(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / (std + 1e-7)

