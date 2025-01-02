from typing import Callable, NoReturn

import numpy as np

from .utils import ActivationName


class Activation:
    def __init__(self, activation: ActivationName) -> None:
        self.__yHat: np.ndarray
        self.__activation: ActivationName = activation

        activations: dict[ActivationName, Callable] = {
            "linear": self.__linear,
            "relu": self.__relu,
            "sigmoid": self.__sigmoid,
            "softmax": self.__softmax,
        }
        derivatives: dict[ActivationName, Callable] = {
            "linear": self.__linear_derivative,
            "relu": self.__relu_derivative,
            "sigmoid": self.__sigmoid_derivative,
            "softmax": self.__softmax_derivate_error,
        }
        try:
            self.__forward: Callable = activations[activation]
            self.__backward: Callable = derivatives[activation]
        except KeyError:
            raise NotImplementedError(
                f'Not Implemented Activation Function "{activation}"'
            )

    def __linear(self, z: np.ndarray) -> np.ndarray:
        return z

    def __relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        positives = np.clip(z, a_min=0, a_max=None)
        negatives = np.clip(z, a_min=None, a_max=0)
        sigmoid_for_positives = 1 / (1 + np.exp(-positives))
        sigmoid_for_negatives = np.exp(negatives) / (1 + np.exp(negatives))
        return np.where(z > 0, sigmoid_for_positives, sigmoid_for_negatives)

    def __softmax(self, z: np.ndarray) -> np.ndarray:
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __linear_derivative(self) -> int:
        return 1

    def __relu_derivative(self) -> np.ndarray:
        return self.__yHat >= 0

    def __sigmoid_derivative(self) -> np.ndarray:
        return self.__yHat * (1 - self.__yHat)

    def __softmax_derivate_error(self) -> NoReturn:
        raise Exception(
            "Activatio: nsoftmax run under cross_entropy_loss, "
            "not derivative, make sure backward has last=True"
        )

    @property
    def activation(self) -> ActivationName:
        return self.__activation

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.__yHat = self.__forward(z)
        return self.__yHat

    def backward(self) -> np.ndarray:
        return self.__backward()

