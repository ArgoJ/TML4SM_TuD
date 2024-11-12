# %%
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_import import load_data
from src.analytic_potential import get_pinola_kirchhoff_stress


def correct_stress_test(file_path: os.PathLike, eps: float = 1e-3) ->  bool | tuple:
    data = load_data(file_path)
    eps = tf.constant(eps, dtype=tf.float32)

    faulty_P = []
    for F, P_test, _ in data:
        P = get_pinola_kirchhoff_stress(F)
        if tf.reduce_any(tf.abs(P - P_test) > eps):
            faulty_P.append((P, P_test))
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
