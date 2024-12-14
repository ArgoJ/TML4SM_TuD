import tensorflow as tf
import numpy as np


from .models import InputGradFFNN, set_use_output_and_derivative
from .analytic_potential import get_C, get_C_features


def predict_multi_cases_naive(
        model: InputGradFFNN, 
        FPW_tup: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
) -> tuple[dict, dict]:                   
    P_labels = {}
    P_predictions = {}
    for name, (case_F, case_P, case_W) in FPW_tup.items():
        P_labels[name] = case_P.numpy().reshape((-1, 9))
         
        features = get_C_features(case_F)
        P_pred = model.predict(features, verbose=0)
        P_predictions[name] = P_pred.reshape((-1, 9))
    return P_labels, P_predictions


@set_use_output_and_derivative
def predict_multi_cases_PANN(
        model: InputGradFFNN, 
        FPW_tup: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    W_labels = {}
    W_predictions = {}

    P_labels = {}
    P_predictions = {}
    for name, (case_F, case_P, case_W) in FPW_tup.items():
        W_labels[name] = case_W.numpy()
        P_labels[name] = case_P.numpy().reshape((-1, 9))
         
        W_pred, P_pred = model.predict(case_F, verbose=0)
        
        W_predictions[name] = W_pred
        P_predictions[name] = P_pred.reshape((-1, 9))
    return P_labels, W_labels, P_predictions, W_predictions


@set_use_output_and_derivative
def predict_identity_F_PANN(model: InputGradFFNN):
    F_eye = tf.eye(3, 3, batch_shape=[1], dtype=tf.float32)

    W_pred_eye, P_pred_eye  = model.predict(F_eye)

    print(f'P predicted: \n{P_pred_eye}')
    print(f'W predicted: \n{W_pred_eye}')