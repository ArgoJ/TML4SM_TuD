import numpy as np

from typing import Literal


def get_scores(labels: dict[str, np.ndarray], predictions: dict[str, np.ndarray], type: Literal['rmse', 'mse', 'r2']) -> dict[str, np.ndarray]:
    scores = {}

    for load_case in labels:
        true = labels[load_case]
        preds = predictions[load_case]

        if type == 'rmse':
            score = _rmse_score(true, preds)
        elif type == 'mse':
            score = _mse_score(true, preds)
        elif type == 'r2':
            score = _r2_score(true, preds)
        else:
            raise NotImplementedError(f'{type} score is not implemented yet!')
        
        scores[load_case] = score.reshape((3, 3))
    
    return scores


def _r2_score(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true_mean = np.mean(true, axis=0)
    ss_total = np.sum((true - true_mean) ** 2, axis=0)
    ss_residual = np.sum((true - pred) ** 2, axis=0)
    r2_score = 1 - (ss_residual / ss_total)
    return np.nan_to_num(r2_score, copy=False, nan=0)

def _rmse_score(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(true - pred), axis=0))

def _mse_score(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square(true - pred), axis=0)

