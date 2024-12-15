import tensorflow as tf

from .models import InputGradFFNN, set_use_output_and_derivative

def are_rotation_matrices(Q: tf.Tensor, eps: float = 1e-6):
    Q = tf.cast(Q, tf.float64) 
    batch_size, n, _ = Q.shape
    I = tf.eye(n, batch_shape=[batch_size], dtype=Q.dtype)
    Q_transpose = tf.transpose(Q, perm=[0, 2, 1])
    Q_transpose_Q = tf.matmul(Q_transpose, Q)
    orthogonal = tf.reduce_all(tf.abs(Q_transpose_Q - I) < eps, axis=[1, 2])
    det_Q = tf.linalg.det(Q)
    determinant_one = tf.abs(det_Q - 1.0) < eps
    return tf.logical_and(orthogonal, determinant_one)


def generate_random_rotation_matrix(batch_size: int, seed: int | None = None) -> tf.Tensor:
    if seed is not None:
        tf.random.set_seed(seed)
    random_matrices = tf.random.normal((batch_size, 3, 3))
    q, _ = tf.linalg.qr(random_matrices)
    rot_mat_mask = are_rotation_matrices(q, eps=1e-7)

    if tf.reduce_all(rot_mat_mask):
        return q
    else:
        valid_q = tf.boolean_mask(q, rot_mat_mask)
        new_q = generate_random_rotation_matrix(tf.math.count_nonzero(~rot_mat_mask))
        return tf.concat([valid_q, new_q], axis=0)


def generate_augmented_dataset(
        data: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]], 
        n_observers: int = 10
) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    Q_batch = generate_random_rotation_matrix(n_observers)
    augmented_data = {}
    for name, (F, P, W) in data.items():
        if not is_positive_definite(F):
            print(f'Not a positive definite F provided in {name}!')
        for idx_obs, Q in enumerate(Q_batch):
            QF = Q * F
            QP = Q * P
            augmented_data[f'{name}_obs{idx_obs}'] = (QF, QP, W)
    return augmented_data

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