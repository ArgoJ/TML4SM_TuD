# %%
import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Callable, Literal

from .analytic_potential import get_pinola_kirchhoff_stress, get_hyperelastic_potential, get_C_features


# %%
def load_df(path: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(path, sep='\s+', header=None)


def load_data(path: os.PathLike) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    df = load_df(path)
    F_batch, P_batch, W_batch = [], [], []
    for _, row in df.iterrows():
        F = tf.constant(row.iloc[:9].to_numpy(), dtype=tf.float32, shape=[3, 3])
        P = tf.constant(row.iloc[9:18].to_numpy(), dtype=tf.float32, shape=[3, 3])
        W = tf.constant(row.iloc[-1:].to_numpy(), dtype=tf.float32, shape=[1, ])
        F_batch.append(F)
        P_batch.append(P)
        W_batch.append(W)

    F_batch = tf.stack(F_batch)  # Shape: (batch, 3, 3)
    P_batch = tf.stack(P_batch)  # Shape: (batch, 3, 3)
    W_batch = tf.stack(W_batch)  # Shape: (batch, 1)
    return F_batch, P_batch, W_batch


def load_concentric(path: os.PathLike) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    df = load_df(path)
    F_batch = []
    for _, F in df.iterrows():
        F = tf.constant(F.to_numpy().reshape((3, 3)), dtype=tf.float32)
        F_batch.append(F)

    F_batch = tf.stack(F_batch)  # Shape: (batch, 3, 3)
    P_batch = get_pinola_kirchhoff_stress(F_batch)
    W_batch = get_hyperelastic_potential(F_batch)
    return F_batch, P_batch, W_batch
    

def load_invariants(path: os.PathLike) -> tf.Tensor:
    df = load_df(path)
    data = df.to_numpy(dtype=np.float32)
    return tf.convert_to_tensor(data, dtype=tf.float32)


def get_naive_dataset(data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    F, P, _ = data
    features = get_C_features(F)
    labels = tf.reshape(P, (-1, 9))
    return features, labels


def get_pann_dataset(data: tuple[tf.Tensor, tf.Tensor, tf.Tensor], which_label: Literal['W', 'P', 'WP'] = 'P') -> tuple[tf.Tensor, tf.Tensor]:
    F, P, W = data
    features = F
    if which_label == 'P':
        labels = P
    elif which_label == 'W':
        labels = W
    elif which_label == 'WP':
        labels = (W, P)
    return features, labels


def get_train_dataset(
        data: dict[str, tuple], 
        preprocess_data_func: Callable[[tuple[tf.Tensor, tf.Tensor, tf.Tensor]], tuple[tf.Tensor, tf.Tensor]], 
        keys: list[str]
) -> tuple[tf.Tensor, tf.Tensor]:
    if len(keys) == 1:
        return preprocess_data_func(data[keys[0]])
    
    new_data = {key: preprocess_data_func(tup) for key, tup in data.items() if key in keys}
    features = tf.concat([new_data[key][0] for key in keys], axis=0)

    label_len = len(next(iter(new_data.values()))[1])
    if label_len > 1:
        labels = [tf.concat([new_data[key][1][i] for key in keys], axis=0) for i in range(label_len)]
    else:
        labels = tf.concat([new_data[key][1] for key in keys], axis=0)
    return features, labels


def load_train_test_concentric(
        path: os.PathLike, 
        test_size: float = 0.2
) -> tuple[dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]], dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]]:
    files = os.listdir(path)
    indices = np.random.permutation(np.arange(len(files)))

    split_index = int(len(indices) * test_size)
    test_indices, train_indices = np.split(indices, [split_index]) 

    train_cases = {}
    for idx in train_indices:
        train_cases[idx] = load_concentric(os.path.join(path, files[idx]))

    test_cases = {}
    for idx in test_indices:
        test_cases[idx] = load_concentric(os.path.join(path, files[idx]))
    
    return train_cases, test_cases 


# %%

def main() -> None:
    calibration_dir = os.path.abspath(os.path.join('hyperelasticity', 'calibration'))
    biaxial_path = os.path.join(calibration_dir, 'biaxial.txt')
    pure_shear_path = os.path.join(calibration_dir, 'pure_shear.txt')
    uniaxial_path = os.path.join(calibration_dir, 'uniaxial.txt')

    biaxial_data = load_data(biaxial_path)
    print(f'Last biaxaial: \n{biaxial_data[2].shape}')

    pure_shear_data = load_data(pure_shear_path)
    print(f'Last pure shear: \n{pure_shear_data[2].shape}')

    uniaxial_data = load_data(uniaxial_path)
    print(f'Last uniaxaial: \n{uniaxial_data[2].shape}')

if __name__ == '__main__':
    main()