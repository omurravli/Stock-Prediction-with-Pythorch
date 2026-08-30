import numpy as np
import pandas as pd


def calculate_log_returns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe[
        ["date", "price"]
    ].copy()

    result["previous_price"] = (
        result["price"].shift(1)
    )

    result["log_return"] = np.log(
        result["price"]
        / result["previous_price"]
    )

    return (
        result
        .dropna()
        .reset_index(drop=True)
    )


def create_sequences(
    values: np.ndarray,
    lookback: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sequences = []
    targets = []
    target_indices = []

    for target_index in range(
        lookback,
        len(values),
    ):
        start_index = (
            target_index - lookback
        )

        sequences.append(
            values[start_index:target_index]
        )

        targets.append(
            values[target_index]
        )

        target_indices.append(
            target_index
        )

    X = np.asarray(
        sequences,
        dtype=np.float32,
    ).reshape(
        -1,
        lookback,
        1,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    return X, y, indices


def reconstruct_prices(
    previous_prices: np.ndarray,
    predicted_returns: np.ndarray,
) -> np.ndarray:
    return (
        previous_prices
        * np.exp(predicted_returns)
    )
