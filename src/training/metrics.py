import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def p90_abs_error(y_true, y_pred) -> float:
    return float(np.percentile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), 90))


def all_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}mae": mae(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}p90_abs_error": p90_abs_error(y_true, y_pred),
    }
