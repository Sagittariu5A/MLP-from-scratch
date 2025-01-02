from typing import Callable

import numpy as np

from .utils import Metrics, LossName


class Loss:
    def __init__(self, loss: LossName) -> None:
        loss_derivative: dict[LossName, Callable] = {
            "binarycrossentropy": self.__binary_cross_entropy_derivative,
            "crossentropy": self.__cross_entropy_derivative,
            "mse": self.__mse_derivative,
        }
        loss_function: dict[LossName, Callable] = {
            "binarycrossentropy": self.__binary_cross_entropy_loss,
            "crossentropy": self.__cross_entropy_loss,
            "mse": self.__mse_loss,
        }
        metrics: dict[LossName, Callable] = {
            "binarycrossentropy": self.__binary_classification_metrics,
            "crossentropy": self.__multiclass_metrics,
            "mse": self.__mse_metrics,
        }
        try:
            self.__loss: LossName = loss
            self.__loss_derivative: Callable = loss_derivative[self.__loss]
            self.__loss_function: Callable = loss_function[self.__loss]
            self.__metrics: Callable = metrics[self.__loss]
        except KeyError:
            raise NotImplementedError(f"Not Implemented Loss Function `{loss}'")

    def __mse_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
    ) -> np.ndarray:
        return (yHat - y) / batch_size

    def __binary_cross_entropy_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
    ) -> np.ndarray:
        return (yHat - y) / batch_size

    def __cross_entropy_derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
    ) -> np.ndarray:
        return (yHat - y) / batch_size

    def __mse_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
        loss = np.power(yHat - y, 2)
        loss = float(loss.mean())
        return loss

    def __binary_cross_entropy_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
        yHat = np.clip(yHat, 1e-7, 1 - 1e-7)
        loss = y * np.log(yHat) + (1 - y) * np.log(1 - yHat)
        loss = float(loss.mean())
        return -1 * loss

    def __cross_entropy_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
        yHat = np.clip(yHat, 1e-7, 1)
        loss = y * np.log(yHat)
        loss = np.sum(loss, axis=1)
        loss = float(loss.mean())
        return -1 * loss

    def __mse_metrics(self, y: np.ndarray, yHat: np.ndarray) -> Metrics:
        y_yHat = y - yHat
        mse = np.power(y_yHat, 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(y_yHat).mean()
        return float(mse), float(rmse), float(mae), 0

    def __binary_classification_metrics(self, y: np.ndarray, yHat: np.ndarray) -> Metrics:
        yHat = yHat > 0.5
        y1, y0, yHat1, yHat0 = y == 1, y == 0, yHat == 1, yHat == 0
        tp = np.sum((y1) & (yHat1))
        fn = np.sum((y1) & (yHat0))
        fp = np.sum((y0) & (yHat1))
        tn = np.sum((y0) & (yHat0))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-7)
        return float(accuracy), float(precision), float(recall), float(f1_score)

    def __multiclass_metrics(self, y: np.ndarray, yHat: np.ndarray) -> Metrics:
        n_class = y.shape[1]
        conf_matrix = np.zeros((n_class, n_class), dtype=int)
        true_labels = y.argmax(axis=1)
        pred_labels = yHat.argmax(axis=1)
        for t, p in zip(true_labels, pred_labels):
            conf_matrix[t, p] += 1
        diagonal = conf_matrix.diagonal()
        tp = diagonal
        fp = conf_matrix.sum(axis=0) - diagonal
        # fn = conf_matrix.sum(axis=1) - diagonal
        accuracy = tp.sum() / conf_matrix.sum()
        precision = tp.sum() / (tp.sum() + fp.sum() + 1e-7)
        return float(accuracy), float(precision), 0, 0

    @property
    def loss_name(self) -> LossName:
        return self.__loss

    def derivative(
        self, y: np.ndarray, yHat: np.ndarray, batch_size: int
    ) -> np.ndarray:
        return self.__loss_derivative(y, yHat, batch_size)

    def loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
        return self.__loss_function(y, yHat)

    def metrics(self, y: np.ndarray, yHat: np.ndarray) -> Metrics:
        return self.__metrics(y, yHat)

