import numpy as np
import tensorflow as tf

from typing import Literal



def strings_containing_substrings(string_list: list[str], substring_list: list[str]) -> list[str]:
    return [string for substring in substring_list for string in string_list if substring.lower() in string.lower()]


def get_scores(
        labels: dict[str, np.ndarray], 
        predictions: dict[str, np.ndarray], 
        type: Literal['rmse', 'mse', 'r2', 'wmae'],
        use_total: bool = False
) -> dict[str, np.ndarray]:
    scores = {}

    for load_case in labels:
        true = labels[load_case]
        preds = predictions[load_case]

        if type == 'rmse':
            score = _rmse_score(true, preds, use_total)
        elif type == 'mse':
            score = _mse_score(true, preds, use_total)
        elif type == 'r2':
            score = _r2_score(true, preds, use_total)
        elif type == 'wmae':
            score = _weighted_mae(true, preds, use_total)
        else:
            raise NotImplementedError(f'{type} score is not implemented yet!')
        
        if not use_total:
            score = score.reshape((3, 3))
        scores[load_case] = score
    
    return scores


def _r2_score(true: np.ndarray, pred: np.ndarray, use_total: bool = False) -> np.ndarray:
    if use_total:
        true_mean = np.mean(true)
        ss_total = np.sum((true - true_mean) ** 2)
        ss_residual = np.sum((true - pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
    else:
        true_mean = np.mean(true, axis=0)
        ss_total = np.sum((true - true_mean) ** 2, axis=0)
        ss_residual = np.sum((true - pred) ** 2, axis=0)
        r2 = 1 - (ss_residual / ss_total)

    return np.nan_to_num(r2, copy=False, nan=0)


def _rmse_score(true: np.ndarray, pred: np.ndarray, use_total: bool=False) -> np.ndarray:
    return np.sqrt(_mse_score(true, pred, use_total))

def _mse_score(true: np.ndarray, pred: np.ndarray, use_total: bool=False) -> np.ndarray:
    squared_errors = np.square(true - pred)
    if use_total:
        return np.mean(squared_errors)
    else:
        return np.mean(squared_errors, axis=0)

def _weighted_mae(true: np.ndarray, pred: np.ndarray, use_total: bool=False) -> np.ndarray:
    errors = np.abs(pred - true)
    weights = 1 / true
    weighted_errors = errors * weights
    if use_total:
        return np.mean(weighted_errors) 
    else:
        return np.mean(weighted_errors, axis=0)