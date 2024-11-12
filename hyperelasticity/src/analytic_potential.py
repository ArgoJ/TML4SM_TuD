# %%
import tensorflow as tf

def get_invariants(F: tf.Tensor) -> tf.Tensor:
    G_ti = tf.constant([[4, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype=tf.float32)
    C = tf.transpose(F) @ F
    i3 = tf.linalg.det(C)
    Cof_C = i3 * tf.linalg.inv(C)

    i1 = tf.linalg.trace(C)
    j =  tf.linalg.det(F)
    i4 = tf.linalg.trace(C @ G_ti)
    i5 = tf.linalg.trace(Cof_C @ G_ti)

    return tf.stack((i1, j, i4, i5))


def get_hyperelastic_potential(F: tf.Tensor) -> tf.Tensor:
    invariants = get_invariants(F)
    i1 = invariants[0]
    j = invariants[1]
    i4 = invariants[2]
    i5 = invariants[3]
    return 8*i1 + 10*j**2 - 56*tf.math.log(j) + 0.2*(i4**2 + i5**2) - 44


def get_pinola_kirchhoff_stress(F: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as g:
        g.watch(F)
        W = get_hyperelastic_potential(F)
    del_W__del_F = g.gradient(W, F)
    return del_W__del_F

# %%
def main() -> None:
    F = tf.eye(3, 3)
    invs = get_invariants(F)
    W = get_hyperelastic_potential(F)
    P= get_pinola_kirchhoff_stress(F)
    print(invs)
    print(W)
    print(P)



if __name__ == '__main__':
    main()
