# %%
import tensorflow as tf


def get_C(F: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matmul(tf.transpose(F, perm=[0, 2, 1]), F, transpose_b=False)


def get_C_features(F: tf.Tensor) -> tf.Tensor:
    C = get_C(F)
    return tf.stack([
        C[:, 0, 0], 
        C[:, 1, 1],
        C[:, 2, 2],
        C[:, 0, 1],
        C[:, 0, 2],
        C[:, 1, 2],
    ], axis=1)


def get_transversely_isotropic_invariants(F: tf.Tensor) -> tf.Tensor:
    # G_ti is the same for all samples
    G_ti = tf.constant([[4, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype=tf.float32)
    C = get_C(F)
    Cof_C = cofactor(C)
    i1 = tf.linalg.trace(C)
    j = tf.linalg.det(F)
    i4 = tf.linalg.trace(tf.linalg.matmul(C, G_ti))
    i5 = tf.linalg.trace(tf.linalg.matmul(Cof_C, G_ti))
    return tf.stack((i1, j, i4, i5), axis=1)


def get_cubic_anisotropic_invariants(F: tf.Tensor) -> tf.Tensor:
    C = get_C(F)
    Cof_C = cofactor(C)
    i1 = tf.linalg.trace(C)
    i2 = tf.linalg.trace(Cof_C)
    j = tf.linalg.det(F)

    C_diag = tf.linalg.diag_part(C) 
    Cof_C_diag = tf.linalg.diag_part(Cof_C) 
    i7 = tf.reduce_sum(tf.square(C_diag), axis=1)
    i11 = tf.reduce_sum(tf.square(Cof_C_diag), axis=1)
    return tf.stack((i1, i2, j, i7, i11), axis=1)


def get_polyconvex_inputs(F: tf.Tensor) -> tf.Tensor:
    det_F = tf.linalg.det(F)
    Cof_F = det_F[:, None, None] * tf.linalg.inv(F)
    return tf.stack((F, Cof_F, det_F), axis=1)


def cofactor(M: tf.Tensor) -> tf.Tensor:
    det_M = tf.linalg.det(M)
    return det_M[:, None, None] * tf.linalg.inv(M)


def get_hyperelastic_potential(F: tf.Tensor) -> tf.Tensor:
    invariants = get_transversely_isotropic_invariants(F)

    i1 = invariants[:, 0]
    j = invariants[:, 1]
    i4 = invariants[:, 2]
    i5 = invariants[:, 3]
    W = 8 * i1 + 10 * j**2 - 56 * tf.math.log(j) + 0.2 * (i4**2 + i5**2) - 44
    return tf.expand_dims(W, axis=-1)


def get_pinola_kirchhoff_stress(F: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape(persistent=True) as g:
        g.watch(F) 
        W = get_hyperelastic_potential(F)

    del_W__del_F = g.gradient(W, F)
    return del_W__del_F


# %%
def main() -> None:
    F = tf.eye(3, 3)
    invs = get_transversely_isotropic_invariants(F)
    W = get_hyperelastic_potential(F)
    P= get_pinola_kirchhoff_stress(F)
    print(invs)
    print(W)
    print(P)



if __name__ == '__main__':
    main()
