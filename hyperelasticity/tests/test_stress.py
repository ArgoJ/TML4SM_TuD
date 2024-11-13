# %%
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_import import load_data
from src.analytic_potential import get_pinola_kirchhoff_stress


def correct_stress_test(file_path: os.PathLike, eps: float = 1e-4) ->  bool | tuple:
    F, P_test, _ = load_data(file_path)
    P = get_pinola_kirchhoff_stress(F)

    eps = tf.constant(eps, dtype=tf.float32)
    diff = tf.abs(P - P_test)  # Shape: (batch_size, 3, 3)
    faulty_mask = tf.reduce_any(diff > eps, axis=[1, 2])

    faulty_P = []
    if tf.reduce_any(faulty_mask):
        faulty_indices = tf.where(faulty_mask)[:, 0]  # Indices of faulty samples
        for idx in faulty_indices:
            faulty_P.append((P[idx].numpy(), P_test[idx].numpy()))

    return not faulty_P, faulty_P


# %%
def main():
    # Data
    calibration_dir = os.path.abspath(os.path.join('hyperelasticity', 'calibration'))
    biaxial_path = os.path.join(calibration_dir, 'biaxial.txt')
    pure_shear_path = os.path.join(calibration_dir, 'pure_shear.txt')
    uniaxial_path = os.path.join(calibration_dir, 'uniaxial.txt')

    biaxial_stress = correct_stress_test(biaxial_path)
    pure_shear_stress = correct_stress_test(pure_shear_path)
    uniaxial_stress = correct_stress_test(uniaxial_path)

    print('='*50)
    print('Stress:')
    print(f'\tBiaxial: {biaxial_stress[0]}') 
    print(f'\tPure Shear: {pure_shear_stress[0]}')
    print(f'\tUniaxial: {uniaxial_stress[0]}')


if __name__ == '__main__':
    main()
