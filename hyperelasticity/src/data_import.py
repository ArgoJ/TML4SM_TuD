# %%
import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

from .analytic_potential import get_C


# %%
def load_df(path: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(path, sep=' ', header=None)

def load_data(path: os.PathLike) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    df = load_df(path)

    F_batch = []
    P_batch = []
    W_batch = []
    for _, row in df.iterrows():
        F = tf.constant(row.iloc[:9].to_numpy().reshape((3, 3)), dtype=tf.float32)
        P = tf.constant(row.iloc[9:-1].to_numpy().reshape((3, 3)), dtype=tf.float32)
        W = tf.constant(row.iloc[-1:].to_numpy(), dtype=tf.float32)
        F_batch.append(F)
        P_batch.append(P)
        W_batch.append(W)

    F_batch = tf.stack(F_batch)  # Shape: (batch, 3, 3)
    P_batch = tf.stack(P_batch)  # Shape: (batch, 3, 3)
    W_batch = tf.stack(W_batch)  # Shape: (batch, 1)

    return F_batch, P_batch, W_batch


def load_invariants(path: os.PathLike) -> tf.Tensor:
    df = load_df(path)
    data = df.to_numpy(dtype=np.float32)
    return tf.convert_to_tensor(data, dtype=tf.float32)


def load_naive_dataset(data_path: os.PathLike) -> tuple[tf.Tensor, tf.Tensor]:
    F, P, _ = load_data(data_path)
    C = get_C(F)
    features = tf.stack([
        C[:, 0, 0], 
        C[:, 1, 1],
        C[:, 2, 2],
        C[:, 0, 1],
        C[:, 0, 2],
        C[:, 1, 2],
    ], axis=1)
    labels = tf.reshape(P, (-1, 9))
    return features, labels


def load_paml_dataset(data_path: os.PathLike) -> tuple[tf.Tensor, tf.Tensor]:
    F, P, W = load_data(data_path)
    features = F
    labels = tf.reshape(P, (-1, 9))
    return features, labels



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