import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    mse = mean_squared_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(mse)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
    }


def direction_accuracy(
    previous_prices: np.ndarray,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
) -> float:
    actual_direction = np.sign(
        actual_prices
        - previous_prices
    )

    predicted_direction = np.sign(
        predicted_prices
        - previous_prices
    )

    correct_predictions = (
        actual_direction
        == predicted_direction
    )

    return float(
        correct_predictions.mean() * 100
    )
