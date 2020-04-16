import os
import math
import numpy as np
import tensorflow as tf
from keras_htr.models.cnn_1drnn_ctc import decode_greedy, beam_search_decode
from keras_htr.edit_distance import compute_cer

from keras_htr.models.base import compute_output_shape


def codes_to_string(codes, char_table):
    return ''.join([char_table.get_character(code) for code in codes])


def get_meta_info(path='lines_dataset/train'):
    import json
    meta_path = os.path.join(path, 'meta.json')
    with open(meta_path) as f:
        s = f.read()

    meta_info = json.loads(s)
    return meta_info


def get_meta_attribute(path, attr_name):
    return get_meta_info(path)[attr_name]


def get_max_image_height(root_path):
    train = os.path.join(root_path, 'train')
    val = os.path.join(root_path, 'validation')
    test = os.path.join(root_path, 'test')

    train_height = get_meta_attribute(train, 'max_height')
    val_height = get_meta_attribute(val, 'max_height')
    test_height = get_meta_attribute(test, 'max_height')

    return max(train_height, val_height, test_height)


def predict_labels(model, X, input_lengths, beam_search=False):
    densities = model.predict(X)

    if beam_search:
        predicted_labels = beam_search_decode(densities, input_lengths)
    else:
        predicted_labels = decode_greedy(densities, input_lengths)
    return predicted_labels.tolist()


def make_true_labels(labels, label_lengths):
    flat_label_lengths = label_lengths.flatten()
    expected_labels = []

    for i, labelling in enumerate(labels):
        labelling_len = flat_label_lengths[i]
        expected_labels.append(labelling.tolist()[:labelling_len])

    return expected_labels


def cer_on_batch(model, batch):
    X, labels, input_lengths, label_lengths = batch

    predicted_labels = model.predict(X, input_lengths=input_lengths).tolist()

    expected_labels = make_true_labels(labels, label_lengths)

    label_error_rates = compute_cer(expected_labels, predicted_labels)

    return tf.reduce_mean(label_error_rates).numpy().flatten()[0]


class LEREvaluator:
    def __init__(self, model, gen, steps, char_table):
        self._model = model
        self._gen = gen
        self._steps = steps or 10
        self._char_table = char_table

    def evaluate(self):
        scores = []

        adapter = self._model.get_adapter()
        for i, example in enumerate(self._gen):
            if i > self._steps:
                break

            image_path, ground_true_text = example
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")

            expected_labels = [[self._char_table.get_label(ch) for ch in ground_true_text]]
            inputs = adapter.adapt_x(image)

            predictions = self._model.predict(inputs)
            cer = compute_cer(expected_labels, predictions.tolist())[0]
            scores.append(cer)

        return np.array(scores).mean()


def get_bin_end(signal, s):
    for i in range(s, len(signal)):
        if not signal[i]:
            return i

    return len(signal) - 1


def get_next_bin(signal, s, min_size=30):
    for i in range(s, len(signal)):
        if signal[i]:
            end_point = get_bin_end(signal, i + 1)
            if end_point - i > min_size:
                return i, end_point

    return None


def line_segmentation(a, threshold=5):
    densities = np.mean(a, axis=1)

    binary_signal = (densities > threshold)

    lines = []
    last_end = 0
    while True:
        ivl = get_next_bin(binary_signal, last_end)
        if ivl is None:
            break

        i, j = ivl

        delta = 15
        slice_from = max(0, i - delta)
        slice_to = min(j + delta, len(a))
        lines.append(a[slice_from:slice_to, :])

        last_end = j + 1

    return lines


def recognize_line(model, image_path, image_height, char_table):
    from keras_htr.preprocessing import prepare_x
    x = prepare_x(image_path, image_height)
    lstm_input_shape = compute_output_shape(x.shape)
    width, channels = lstm_input_shape

    X = np.stack([x])
    input_lengths = np.array(width).reshape(1, 1)
    labellings = predict_labels(model, X, input_lengths)

    labels = labellings[0]
    s = ''.join([char_table.get_character(code) for code in labels])
    return s


def recognize_document(model, image_path, image_height, char_table):
    from keras_htr.preprocessing import binarize

    img = tf.keras.preprocessing.image.load_img(image_path)
    a = tf.keras.preprocessing.image.img_to_array(img)
    a = binarize(a)
    line_images = line_segmentation(a)

    lines = []
    for image_array in line_images:

        image = tf.keras.preprocessing.image.array_to_img(255.0 - image_array)
        image.save('tmp.png')
        s = recognize_line(model, 'tmp.png', image_height, char_table)
        lines.append(s)

    return '\n'.join(lines)
