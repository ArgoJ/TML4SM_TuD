import tensorflow as tf

from .models import InputGradFFNN, set_use_output_and_derivative


def generate_random_rotation_matrix(batch_size: int, seed: int | None = None) -> tf.Tensor:
    if seed is not None:
        tf.random.set_seed(seed)
    random_matrices = tf.random.normal((batch_size, 3, 3))
    orthogonal_matrices = []
    for i in range(batch_size):
        q, _ = tf.linalg.qr(random_matrices[i])
        orthogonal_matrices.append(q)
    return tf.stack(orthogonal_matrices, axis=0)


def generate_positive_definite_matrix(batch_size: int, seed: int | None = None) -> tf.Tensor:
    if seed is not None:
        tf.random.set_seed(seed)
    A = tf.random.normal((batch_size, 3, 3))
    symmetric_matrix = tf.matmul(A, A, transpose_b=True)
    identity_matrix = tf.eye(3, batch_shape=[batch_size])
    positive_definite_matrix = symmetric_matrix + 1e-6 * identity_matrix
    return positive_definite_matrix


def is_positive_definite(F: tf.Tensor) -> tf.Tensor:
    eigenvalues = tf.linalg.eigvals(F)
    return tf.reduce_all(tf.math.real(eigenvalues) > 0)


@set_use_output_and_derivative
def check_objectivity_condition(
        model: InputGradFFNN, 
        F: tf.Tensor,  
        n_observers: int = 10, 
        eps: float = 1e-6,
        seed: int | None = None
) -> bool:
    if not is_positive_definite(F):
        print('Not a positive definite F provided, created new random F!')
        F = generate_positive_definite_matrix(F.shape[0], seed)
    
    Q_matrices = generate_random_rotation_matrix(n_observers, seed)
    W_F, P_F = model(F)

    for Q in Q_matrices:
        QF = Q * F
        W_QF, P_QF = model(QF)

        if not tf.reduce_all(tf.abs(W_F - W_QF) < eps) or \
                not tf.reduce_all(tf.abs(P_F - P_QF) < eps):
            return False
    return True