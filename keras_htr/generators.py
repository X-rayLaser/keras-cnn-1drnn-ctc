import os
import random
import tensorflow as tf
import numpy as np
import json
import scipy
from keras_htr.models import compute_output_shape


class BaseGenerator:
    def __iter__(self):
        raise NotImplementedError


def get_dictionary():
    dictionary = []
    with open('words_dataset/dictionary.txt') as f:
        for i, line in enumerate(f.readlines()):
            dictionary.append(line.rstrip())
    return dictionary


class CompiledDataset:
    def __init__(self, dataset_root):
        self._root = dataset_root
        self._lines = []

        lines_path = os.path.join(dataset_root, 'lines.txt')
        with open(lines_path) as f:
            for row in f.readlines():
                self._lines.append(row.rstrip('\n'))

        meta_path = os.path.join(dataset_root, 'meta.json')
        with open(meta_path) as f:
            s = f.read()
        meta_info = json.loads(s)

        self.__dict__.update(meta_info)

        self._num_examples = meta_info['num_examples']

        os.path.dirname(dataset_root)

    @property
    def size(self):
        return self._num_examples

    def __iter__(self):
        for i in range(self._num_examples):
            yield self.get_example(i)

    def get_example(self, line_index):
        text = self._lines[line_index]
        image_path = os.path.join(self._root, str(line_index) + '.png')
        return image_path, text


class LinesGenerator(BaseGenerator):
    def __init__(self, dataset_root, char_table, batch_size=4, augment=False, batch_adapter=None):
        self._root = dataset_root
        self._char_table = char_table
        self._batch_size = batch_size
        self._augment = augment

        if batch_adapter is None:
            self._adapter = CTCAdapter()
        else:
            self._adapter = batch_adapter

        self._ds = CompiledDataset(dataset_root)

        self._indices = list(range(self._ds.size))

    @property
    def size(self):
        return self._ds.size

    def __iter__(self):
        batches_gen = self.get_batches()
        self._adapter.fit(batches_gen)

        while True:
            for batch in self.get_batches():
                yield self._adapter.adapt_batch(batch)

    def get_batches(self):
        random.shuffle(self._indices)
        image_arrays = []
        labellings = []
        for line_index in self._indices:
            image_array, labels = self.get_example(line_index)
            image_arrays.append(image_array)
            labellings.append(labels)

            if len(labellings) >= self._batch_size:
                batch = image_arrays, labellings
                image_arrays = []
                labellings = []
                yield batch

        if len(labellings) >= 1:
            yield image_arrays, labellings

    def text_to_class_labels(self, text):
        return [self._char_table.get_label(ch) for ch in text]

    def get_example(self, line_index):
        image_path, text = self._ds.get_example(line_index)
        img = tf.keras.preprocessing.image.load_img(image_path, grayscale=True)
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = a / 255.0
        y = self.text_to_class_labels(text)
        return x, y


class BatchAdapter:
    def fit(self, batches):
        pass

    def adapt_batch(self, batch):
        raise NotImplementedError

    def _pad_labellings(self, labellings, target_length, padding_code=0):
        padded_labellings = []
        for labels in labellings:
            padding_size = target_length - len(labels)
            assert padding_code >= 0
            new_labelling = labels + [padding_code] * padding_size
            assert len(new_labelling) > 0
            padded_labellings.append(new_labelling)

        return padded_labellings

    def _pad_array_width(self, a, target_width):
        width = a.shape[1]

        right_padding = target_width - width

        assert right_padding >= 0

        horizontal_padding = (0, right_padding)
        vertical_padding = (0, 0)
        depth_padding = (0, 0)
        return scipy.pad(a, pad_width=[vertical_padding, horizontal_padding, depth_padding])

    def _pad_image_arrays(self, image_arrays, target_width):
        return [self._pad_array_width(a, target_width) for a in image_arrays]


class CTCAdapter(BatchAdapter):
    def adapt_batch(self, batch):
        image_arrays, labellings = batch

        current_batch_size = len(labellings)

        target_width = max([a.shape[1] for a in image_arrays])
        padded_arrays = self._pad_image_arrays(image_arrays, target_width)

        X = np.array(padded_arrays).reshape(current_batch_size, *padded_arrays[0].shape)

        target_length = max([len(labels) for labels in labellings])
        padded_labellings = self._pad_labellings(labellings, target_length)

        labels = np.array(padded_labellings, dtype=np.int32).reshape(current_batch_size, -1)

        lstm_input_shapes = [compute_output_shape(a.shape) for a in image_arrays]
        widths = [width for width, channels in lstm_input_shapes]
        input_lengths = np.array(widths, dtype=np.int32).reshape(current_batch_size, 1)

        label_lengths = np.array([len(labelling) for labelling in labellings],
                                 dtype=np.int32).reshape(current_batch_size, 1)

        return [X, labels, input_lengths, label_lengths], labels


class ConvolutionalEncoderDecoderAdapter(BatchAdapter):
    def __init__(self, char_table, max_image_width, max_text_length):
        self._sos = char_table.sos
        self._eos = char_table.eos
        self._num_classes = char_table.size

        self._max_image_width = max_image_width
        self._max_text_length = max_text_length

    def fit(self, batches):
        pass

    def adapt_batch(self, batch):
        image_arrays, labellings = batch

        padded_arrays = self._pad_image_arrays(image_arrays, self._max_image_width)
        padded_labellings = self._pad_labellings(labellings, self._max_text_length,
                                                 padding_code=self._eos)

        batch_size = len(labellings)
        x = np.array(padded_arrays).reshape((batch_size, -1, self._max_image_width, 1))

        sos_column = np.ones((batch_size, 1)) * self._sos
        eos_column = np.ones((batch_size, 1)) * self._eos

        decoder_x = np.concatenate([sos_column, padded_labellings], axis=1)[:, :-1]
        decoder_y = np.concatenate([padded_labellings, eos_column], axis=1)[:, 1:]

        decoder_x = tf.keras.utils.to_categorical(decoder_x, num_classes=self._num_classes)
        decoder_y = tf.keras.utils.to_categorical(decoder_y, num_classes=self._num_classes)

        decoder_y = list(np.swapaxes(decoder_y, 0, 1))

        return [x, decoder_x], decoder_y

# todo: consider doing preprocessing during dataset building
# todo: inference for attention model
# todo: factory methods on model classes for creating instances for preprocessors, batch adapters, predictors etc.
