import tensorflow as tf


def to_sparse_tensor(sequences):
    indices = []
    values = []
    max_len = max(len(s) for s in sequences)

    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            indices.append((i, j))
            values.append(sequences[i][j])

    dense_shape = [len(sequences), max_len]
    return tf.SparseTensor(indices, values, dense_shape)


def fill_gaps(labels):
    for y in labels:
        if len(y) == 0:
            y.append(-1)


def compute_edit_distance(y_true, y_pred, normalize=True):
    fill_gaps(y_pred)

    sparse_y_true = to_sparse_tensor(y_true)
    sparse_y_pred = to_sparse_tensor(y_pred)
    return tf.edit_distance(sparse_y_pred, sparse_y_true, normalize=normalize)


def compute_cer(y_true, y_pred):
    distances = compute_edit_distance(y_true, y_pred, normalize=False)
    return normalize_distances(distances, y_true, y_pred)


def normalize_distances(distances, expected_labels, predicted_labels):
    norm_factors = []
    for i, dist in enumerate(distances):
        max_len = max(len(expected_labels[i]), len(predicted_labels[i]))
        norm_factors.append(max_len)

    return tf.divide(tf.dtypes.cast(distances, tf.float32),
                     tf.constant(norm_factors, dtype=tf.float32))
