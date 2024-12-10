# %%
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_import import load_data, load_invariants
from src.analytic_potential import get_transversely_isotropic_invariants


def correct_invariants_test(file_path: os.PathLike, test_file_path: os.PathLike, eps: float = 1e-6) ->  bool | tuple:
    invs_test = load_invariants(test_file_path)

    F, *_ = load_data(file_path)
    invs = get_transversely_isotropic_invariants(F)

    diff = tf.abs(invs - invs_test)
    eps = tf.constant(eps, dtype=tf.float32)
    faulty_mask = tf.reduce_any(diff > eps, axis=-1)

    faulty_tensors = []
    if tf.reduce_any(faulty_mask):
        faulty_indices = tf.where(faulty_mask)[:, 0]  # Indices of faulty samples
        for idx in faulty_indices:
            faulty_tensors.append((invs[idx].numpy(), invs_test[idx].numpy()))

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

    print('='*50)
    print('Invariants:')
    print(f'\tBiaxial: {biaxial_test[0]}') 
    print(f'\tPure Shear: {pure_shear_test[0]}')
    print(f'\tUniaxial: {uniaxial_test[0]}')

if __name__ == '__main__':
    main()


# %%
