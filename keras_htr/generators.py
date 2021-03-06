import os
import random
import tensorflow as tf
import json

from keras_htr.adapters.cnn_1drnn_ctc_adapter import CTCAdapter


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
    def batch_size(self):
        return self._batch_size

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
        img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = a / 255.0
        y = self.text_to_class_labels(text)
        return x, y

# todo: consider doing preprocessing during dataset building
# todo: inference for attention model
# todo: factory methods on model classes for creating instances for preprocessors, batch adapters, predictors etc.
