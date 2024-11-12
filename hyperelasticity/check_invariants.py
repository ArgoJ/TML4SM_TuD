# %%
import os
import tensorflow as tf

from data_import import load_data, load_invariants
from analytic_potential import get_invariants


def correct_invariants_test(file_path: os.PathLike, test_file_path: os.PathLike, eps: float = 1e-6) ->  bool | tuple:
    data = load_data(file_path)
    test_data = load_invariants(test_file_path)

    faulty_tensors = []

    for (F, _, _), invs_test in zip(data, test_data):
        invs = get_invariants(F)

        if tf.reduce_any(tf.abs(invs - invs_test) > tf.constant(eps, dtype=tf.float32)):
            faulty_tensors.append((invs, invs_test))

    return not faulty_tensors, faulty_tensors



# %%
def main():
    # Data
    calibration_dir = os.path.abspath(os.path.join('hyperelasticity', 'calibration'))
    biaxial_path = os.path.join(calibration_dir, 'biaxial.txt')
    pure_shear_path = os.path.join(calibration_dir, 'pure_shear.txt')
    uniaxial_path = os.path.join(calibration_dir, 'uniaxial.txt')

    # Invariants
    invariants_dir = os.path.abspath(os.path.join('hyperelasticity', 'invariants'))
    biaxial_inv_path = os.path.join(invariants_dir, 'I_biaxial.txt')
    pure_shear_inv_path = os.path.join(invariants_dir, 'I_pure_shear.txt')
    uniaxial_inv_path = os.path.join(invariants_dir, 'I_uniaxial.txt')

    biaxial_test = correct_invariants_test(biaxial_path, biaxial_inv_path)
    pure_shear_test = correct_invariants_test(pure_shear_path, pure_shear_inv_path)
    uniaxial_test = correct_invariants_test(uniaxial_path, uniaxial_inv_path)
    print(f'Biaxial: {biaxial_test[0]}') 
    print(f'Pure Shear: {pure_shear_test[0]}')
    print(f'Uniaxial: {uniaxial_test[0]}')

    

if __name__ == '__main__':
    main()


# %%
