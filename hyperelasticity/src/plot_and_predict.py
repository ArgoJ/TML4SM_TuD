import numpy as np
import tensorflow as tf

from .models import CustomFFNN, TransIsoInvariantsICNN, Cubic_Anisotropic_Invariants_Layer, DeformationICNN
from .predict_utils import predict_multi_cases_PANN, predict_multi_cases_naive
from .plots import plot_stress_predictions, plot_stress_correlation, plot_energy_prediction 


def predict_and_plot_naive_model(
        naive_model: CustomFFNN, 
        examples_FPW_tup: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
        suptitle: str = ''
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    suptitle += ' '
    P_naive_test_labels, P_naive_test_preds = predict_multi_cases_naive(naive_model, examples_FPW_tup)
    plot_stress_predictions(P_naive_test_labels, P_naive_test_preds, suptitle=f'{suptitle}Stress $P$')
    plot_stress_correlation(P_naive_test_labels, P_naive_test_preds, suptitle=f'{suptitle}Stress Correlation')
    return P_naive_test_labels, P_naive_test_preds



def predict_and_plot_pann_model(
        pann_model: TransIsoInvariantsICNN | Cubic_Anisotropic_Invariants_Layer | DeformationICNN, 
        examples_FPW_tup: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
        suptitle: str = ''
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    suptitle += ' '
    (
        P_pann_test_labels, W_pann_test_labels, P_pann_test_preds, W_pann_test_preds
    ) = predict_multi_cases_PANN(pann_model, examples_FPW_tup)
    plot_stress_predictions(P_pann_test_labels, P_pann_test_preds, suptitle=f'{suptitle}Stress $P$')
    plot_energy_prediction(W_pann_test_labels, W_pann_test_preds, suptitle=f'{suptitle}Energy $W$')
    plot_stress_correlation(P_pann_test_labels, P_pann_test_preds, suptitle=f'{suptitle}Stress Correlation')
    return P_pann_test_labels, P_pann_test_preds