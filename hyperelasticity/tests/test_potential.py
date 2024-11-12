# %%
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_import import load_data
from src.analytic_potential import get_hyperelastic_potential


def correct_strain_energy_density(file_path: os.PathLike, eps: float = 1e-3) ->  bool | tuple:
    data = load_data(file_path)
    eps = tf.constant(eps, dtype=tf.float32)

    faulty_W = []
    for F, _, W_test in data:
        W = get_hyperelastic_potential(F)
        if tf.abs(W - W_test) > eps:
            faulty_W.append((W, W_test))
    return not faulty_W, faulty_W




# %%
def main():
    # Data
    calibration_dir = os.path.abspath(os.path.join('hyperelasticity', 'calibration'))
    biaxial_path = os.path.join(calibration_dir, 'biaxial.txt')
    pure_shear_path = os.path.join(calibration_dir, 'pure_shear.txt')
    uniaxial_path = os.path.join(calibration_dir, 'uniaxial.txt')

    biaxial_energy = correct_strain_energy_density(biaxial_path)
    pure_shear_energy = correct_strain_energy_density(pure_shear_path)
    uniaxial_energy = correct_strain_energy_density(uniaxial_path)

    print('='*50)
    print('Energy:')
    print(f'\tBiaxial: {biaxial_energy[0]}') 
    print(f'\tPure Shear: {pure_shear_energy[0]}')
    print(f'\tUniaxial: {uniaxial_energy[0]}')


if __name__ == '__main__':
    main()
