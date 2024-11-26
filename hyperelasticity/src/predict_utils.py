import tensorflow as tf
import numpy as np


from .models import InputGradFFNN


def predict_multi_cases(
        model: InputGradFFNN, 
        input_label_tup: dict[str, tuple[tf.Tensor, tf.Tensor]],
        label_is_stress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = {}
    predictions = {}
    for name, (case_inputs, case_labels) in input_label_tup.items():
        if label_is_stress:
            labels[name] = case_labels.numpy().reshape((-1, 9))
            predictions[name] = model.predict(case_inputs).reshape((-1, 9))
        else:
            labels[name] = case_labels
            predictions[name] = model.predict(case_inputs)

    return labels, predictions 



def predict_identity_F(model: InputGradFFNN):
    F_eye = tf.eye(3, 3, batch_shape=[1], dtype=tf.float32)

    if not model.use_output_and_derivative:
        model.use_output_and_derivative = True
        model.compile()

        W_pred_eye, P_pred_eye,  = model.predict(F_eye)
        W_pred_eye = model.predict(F_eye)

        model.use_output_and_derivative = False
        model.compile()

    print(f'P predicted: \n{P_pred_eye}')
    print(f'W predicted: \n{W_pred_eye}')