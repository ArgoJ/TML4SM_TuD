# %%
import os
import tensorflow as tf

from data_import import load_data
from analytic_potential import get_hyperelastic_potential, get_pinola_kirchhoff_stress


def correct_stress_test(file_path: os.PathLike, eps: float = 1e-3) ->  bool | tuple:
    data = load_data(file_path)
    eps = tf.constant(eps, dtype=tf.float32)

    faulty_P = []
    for F, P_test, _ in data:
        P = get_pinola_kirchhoff_stress(F)
        if tf.reduce_any(tf.abs(P - P_test) > eps):
            faulty_P.append((P, P_test))
    return not faulty_P, faulty_P

def correct_strain_energy_density(file_path: os.PathLike, eps: float = 1e-3) ->  bool | tuple:
    data = load_data(file_path)
    eps = tf.constant(eps, dtype=tf.float32)

    faulty_W = []
    for F, P_test, W_test in data:
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

    biaxial_stress = correct_stress_test(biaxial_path)
    pure_shear_stress = correct_stress_test(pure_shear_path)
    uniaxial_stress = correct_stress_test(uniaxial_path)
    print('#'*20)
    print('Stress::')
    print(f'Biaxial: {biaxial_stress[0]}') 
    print(f'Pure Shear: {pure_shear_stress[0]}')
    print(f'Uniaxial: {uniaxial_stress[0]}')

    biaxial_energy = correct_strain_energy_density(biaxial_path)
    pure_shear_energy = correct_strain_energy_density(pure_shear_path)
    uniaxial_energy = correct_strain_energy_density(uniaxial_path)
    print('#'*20)
    print('Energy::')
    print(f'Biaxial: {biaxial_energy[0]}') 
    print(f'Pure Shear: {pure_shear_energy[0]}')
    print(f'Uniaxial: {uniaxial_energy[0]}')

    

if __name__ == '__main__':
    main()
