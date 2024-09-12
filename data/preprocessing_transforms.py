from abc import abstractmethod, ABC
import numpy as np
from segy_preprocessing import (
    zero_time_correction,
    high_and_low_filters,
    iad,
)


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        data, targets = self.transform(data, targets)
        return data, targets


class ZeroTimeCorrection(Transform):
    def __init__(self, correction_parameter: float = 0.05):
        super().__init__()
        self.correction_parameter = correction_parameter

    def transform(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        return zero_time_correction(data, self.correction_parameter, targets)


class RemoveBackground(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        average_trace = np.mean(data, axis=0)
        filtered_data = data - average_trace[np.newaxis, :]
        return filtered_data, targets


class HighPassFilter(Transform):
    def __init__(self, window_width: int):
        super().__init__()
        self.filter_type = "high"
        self.window_width = window_width

    def transform(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        return high_and_low_filters(data, self.filter_type, self.window_width), targets


class LowPassFilter(Transform):
    def __init__(self, window_width: int):
        super().__init__()
        self.filter_type = "low"
        self.window_width = window_width

    def transform(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        return high_and_low_filters(data, self.filter_type, self.window_width), targets


class Iad(Transform):
    def __init__(
        self,
        function: str,
        correction_parameter: float,
        multiplication_factor: int,
        start_weight: int,
        end_weight: int,
    ):
        super().__init__()
        self.function = function
        self.correction_paramter = correction_parameter
        self.multiplication_factor = multiplication_factor
        self.start_weight = start_weight
        self.end_weight = end_weight

    def transform(self, data: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        return (
            iad(
                data,
                self.function,
                self.correction_paramter,
                self.multiplication_factor,
                self.start_weight,
                self.end_weight,
            ),
            targets,
        )
