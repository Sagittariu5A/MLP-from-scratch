from typing import Callable, Any
from abc import ABC, abstractmethod

import numpy as np

from .activations import Activation
from .utils import (
    DenseLayerData,
    DropoutLayerData,
    WeightsInitializerName,
    OptimizerName,
    ActivationName,
    LayerTypeName,
)


class Layer(ABC):
    def __init__(self, *, type: LayerTypeName, output_dim: int) -> None:
        self._type: LayerTypeName = type
        self._output_dim: int = output_dim

    @abstractmethod
    def __call__(self, _: 'Layer') -> None:
        pass

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def set_output_dim(self, output_dim: int) -> None:
        self._output_dim = output_dim

    @property
    def type(self) -> LayerTypeName:
        return self._type


class Input(Layer):
    def __init__(self, input_dim: int) -> None:
        super().__init__(type="input", output_dim=input_dim)

    def __call__(self, _: Layer) -> None:
        pass


class TrainLayer(Layer):
    def __init__(self, *, type: LayerTypeName, output_dim: int) -> None:
        super().__init__(type=type, output_dim=output_dim)

    @abstractmethod
    def backup(self) -> None:
        pass

    @abstractmethod
    def restore(self) -> None:
        pass

    @abstractmethod
    def init(self, *_: Any, **__: Any) -> Any:
        pass

    @abstractmethod
    def save(self) -> DenseLayerData | DropoutLayerData:
        pass

    @abstractmethod
    def load(self, *_: Any, **__: Any) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, *, train: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, gradient: np.ndarray, *, last: bool = False) -> np.ndarray:
        pass


class Dropout(TrainLayer):
    def __init__(self, ratio: float = 0.5, output_dim : int = 0) -> None:
        super().__init__(type="dropout", output_dim=output_dim)
        assert  0.0 < ratio < 1.0 , Exception("Dropout Layer: ratio shall be between 0 and 1 (0 < ratio < 1)")
        self.__ratio: float = ratio
        self.__scalar: float = 1.0 / ( 1.0 - ratio)
        self.__mask: np.ndarray

    def __call__(self, layer: Layer) -> None:
        self.set_output_dim(layer.output_dim)

    def backup(self) -> None:
        pass

    def restore(self) -> None:
        pass

    def init(self, *_: Any, **__: Any) -> Any:
        pass

    def save(self) -> DropoutLayerData:
        return DropoutLayerData(
            self.__ratio,
            self.output_dim,
            self.type
        )

    def load(self, *_: Any, **__: Any) -> None:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x

    def forward(self, x: np.ndarray, *, train: bool = True) -> np.ndarray:
        if train:
            self.__mask = np.random.uniform(0.0, 1.0, x.shape) < self.__ratio
            self.__mask = self.__mask * self.__scalar # multiply by scalar
            return x * self.__mask
        else:
            return x

    def backward(self, gradient: np.ndarray, *, last: bool = False) -> np.ndarray:
        if last:
            raise Exception("Dropout Layer shall not be last layer on model.")
        return gradient * self.__mask


class Dense(TrainLayer):
    def __init__(
        self,
        output_dim: int,
        activation: ActivationName,
        weights_initializer: WeightsInitializerName = "xavier_uniform",
    ) -> None:
        super().__init__(type="dense", output_dim=output_dim)
        self.__optimizer_forward: Callable = self.__gd_forward
        self.__optimizer_backward: Callable
        self.__weights_initializer: WeightsInitializerName = weights_initializer
        self.__activation: Activation = Activation(activation)
        self.__x: np.ndarray
        self.__weights: np.ndarray
        self.__bias: np.ndarray
        self.__weights_backup: np.ndarray
        self.__bias_backup: np.ndarray

    def __gd_init(self) -> None:
        """gd is gradient descent"""
        self.__optimizer_forward = self.__gd_forward
        self.__optimizer_backward = self.__gd_backward

    def __gd_forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.__weights) + self.__bias

    def __gd_backward(self, gradient: np.ndarray) -> np.ndarray:
        self.__weights -= self.__lr * np.dot(self.__x.T, gradient)
        self.__bias -= self.__lr * np.sum(gradient, axis=0)
        return np.dot(gradient, self.__weights.T)

    def __nesterov_momentum_gd_init(self) -> None:
        """Nesterov Gradient Descent"""
        self.__vw: np.ndarray = np.zeros_like(self.__weights)
        self.__vb: np.ndarray = np.zeros_like(self.__bias)
        self.__optimizer_forward = self.__nesterov_momentum_gd_forward
        self.__optimizer_backward = self.__nesterov_momentum_gd_backward

    def __nesterov_momentum_gd_forward(self, x: np.ndarray) -> np.ndarray:
        new_weights = self.__weights - self.__vw
        new_bias = self.__bias - self.__vb
        return np.dot(x, new_weights) + new_bias

    def __nesterov_momentum_gd_backward(self, gradient: np.ndarray) -> np.ndarray:
        beta1 = self.__beta1
        self.__vw = beta1 * self.__vw + self.__lr * np.dot(self.__x.T, gradient)
        self.__vb = beta1 * self.__vb + self.__lr * np.sum(gradient, axis=0)
        self.__weights -= self.__vw
        self.__bias -= self.__vb
        return np.dot(gradient, self.__weights.T)

    def __rmsprop_init(self) -> None:
        self.__sw: np.ndarray = np.zeros_like(self.__weights)
        self.__sb: np.ndarray = np.zeros_like(self.__bias)
        self.__optimizer_forward = self.__rmsprop_forward
        self.__optimizer_backward = self.__rmsprop_backward

    def __rmsprop_forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.__weights) + self.__bias

    def __rmsprop_backward(self, gradient: np.ndarray) -> np.ndarray:
        beta2, epsilon = self.__beta2, 1e-9
        w_gradient = np.dot(self.__x.T, gradient)
        b_gradient = np.sum(gradient, axis=0)
        self.__sw = beta2 * self.__sw + (1 - beta2) * np.square(w_gradient)
        self.__sb = beta2 * self.__sb + (1 - beta2) * np.square(b_gradient)
        self.__weights -= self.__lr * w_gradient / np.sqrt(self.__sw + epsilon)
        self.__bias -= self.__lr * b_gradient / np.sqrt(self.__sb + epsilon)
        return np.dot(gradient, self.__weights.T)

    def __adam_init(self) -> None:
        self.__vw: np.ndarray = np.zeros_like(self.__weights)
        self.__vb: np.ndarray = np.zeros_like(self.__bias)
        self.__sw: np.ndarray = np.zeros_like(self.__weights)
        self.__sb: np.ndarray = np.zeros_like(self.__bias)
        self.__optimizer_forward = self.__adam_forward
        self.__optimizer_backward = self.__adam_backward
        self.__t: int = 0

    def __adam_forward(self, x: np.ndarray) -> np.ndarray:
        # new_weights = self.__weights - self.__vw
        # new_bias = self.__bias - self.__vb
        # return np.dot(x, new_weights) + new_bias
        self.__t += 1
        return np.dot(x, self.__weights) + self.__bias

    def __adam_backward(self, gradient: np.ndarray) -> np.ndarray:
        beta1, beta2, epsilon = self.__beta1, self.__beta2, 1e-7
        w_gradient = np.dot(self.__x.T, gradient)
        b_gradient = np.sum(gradient, axis=0)
        self.__vw = beta1 * self.__vw + (1 - beta1) * w_gradient
        self.__vb = beta1 * self.__vb + (1 - beta1) * b_gradient
        self.__sw = beta2 * self.__sw + (1 - beta2) * np.square(w_gradient)
        self.__sb = beta2 * self.__sb + (1 - beta2) * np.square(b_gradient)
        vw_hat = self.__vw / (1 - beta1 ** self.__t)
        vb_hat = self.__vb / (1 - beta1 ** self.__t)
        sw_hat = self.__sw / (1 - beta2 ** self.__t)
        sb_hat = self.__sb / (1 - beta2 ** self.__t)
        self.__weights -= self.__lr * vw_hat / np.sqrt(sw_hat + epsilon)
        self.__bias -= self.__lr * vb_hat / np.sqrt(sb_hat + epsilon)
        return np.dot(gradient, self.__weights.T)

    def __call__(self, layer: Layer) -> None:
        """
        object call function for initialization: weights, biases, optimizer_init
        take layer for getting 'layer' output_dim as own input_shape
        """
        input_shape = layer.output_dim
        match self.__weights_initializer:
            case "random":
                self.__weights: np.ndarray = np.random.randn(
                    input_shape, self._output_dim
                )
                self.__bias: np.ndarray = np.random.randn(self._output_dim)

            case "uniform":
                max = 1 / np.sqrt(input_shape)
                self.__weights: np.ndarray = np.random.uniform(
                    -max, max, size=(input_shape, self._output_dim)
                )
                self.__bias: np.ndarray = np.random.uniform(
                    -max, max, size=self._output_dim
                )

            case "xavier_uniform":
                max = np.sqrt(6 / (input_shape + self._output_dim))
                self.__weights: np.ndarray = np.random.uniform(
                    -max, max, size=(input_shape, self._output_dim)
                )
                self.__bias: np.ndarray = np.random.uniform(
                    -max, max, size=self._output_dim
                )

            case "xavier_normal":
                std = np.sqrt(2 / (input_shape + self._output_dim))
                self.__weights: np.ndarray = np.random.normal(
                    0, std, size=(input_shape, self._output_dim)
                )
                self.__bias: np.ndarray = np.random.normal(
                    0, std, size=self._output_dim
                )

            case "he_uniform":
                max = np.sqrt(6 / input_shape)
                self.__weights: np.ndarray = np.random.uniform(
                    -max, max, size=(input_shape, self._output_dim)
                )
                self.__bias: np.ndarray = np.random.uniform(
                    -max, max, size=self._output_dim
                )

            case "he_normal":
                std = np.sqrt(2 / input_shape)
                self.__weights: np.ndarray = np.random.normal(
                    0, std, size=(input_shape, self._output_dim)
                )
                self.__bias: np.ndarray = np.random.normal(
                    0, std, size=self._output_dim
                )

            case _:
                raise NotImplementedError(
                    f"DenseLayer __call__ not implemented [{self.__weights_initializer}] parameters initilizer"
                )

    def backup(self) -> None:
        self.__weights_backup = self.__weights.copy()
        self.__bias_backup = self.__bias.copy()

    def restore(self) -> None:
        self.__weights = self.__weights_backup
        self.__bias = self.__bias_backup

    def init(
        self,
        optimizer: OptimizerName,
        lr: float,
        beta2: float,
        beta1: float,
    ) -> None:
        self.__lr: float = lr
        self.__beta1: float = beta1
        self.__beta2: float = beta2
        match optimizer:
            case "gd" | "gradient_descent":
                self.__gd_init()
            case "nag" | "nmgd" | "nesterov_momentum_gradient_descent":
                self.__nesterov_momentum_gd_init()
            case "rmsprop":
                self.__rmsprop_init()
            case "adam":
                self.__adam_init()
            case _:
                raise NotImplementedError(
                    f"DenseLayer init not implemented [{optimizer}] optimizer"
                )

    def save(self) -> DenseLayerData:
        return DenseLayerData(
            self.__weights,
            self.__bias,
            self.__activation.activation,
            self._output_dim,
            self.type
        )

    def load(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.__weights = weights
        self.__bias = bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.__weights) + self.__bias
        return self.__activation.forward(z)

    def forward(self, x: np.ndarray, *, train: bool = True) -> np.ndarray:
        if train:
            self.__x = x
            z = self.__optimizer_forward(x)
        else:
            z = self.__gd_forward(x)
        return self.__activation.forward(z)

    def backward(self, gradient: np.ndarray, *, last: bool = False) -> np.ndarray:
        if not last:
            gradient = gradient * self.__activation.backward()
        return self.__optimizer_backward(gradient)

