import os
import random
import tensorflow as tf
import numpy as np
import json
import scipy
from keras_htr.models import compute_output_shape


def get_zero_padded_array(image_path, target_height):
    img = tf.keras.preprocessing.image.load_img(image_path)
    a = pad_image_height(img, target_height)

    new_height, new_width = a.shape

    if new_width <= target_height:
        img = tf.keras.preprocessing.image.array_to_img(a)
        return pad_image(img, target_height, target_height + 1)

    return a


def get_image_array(image_path, target_height):
    img = tf.keras.preprocessing.image.load_img(image_path)

    aspect_ratio = img.width / img.height

    new_width = int(target_height * aspect_ratio)
    if new_width <= target_height:
        return pad_image(img, target_height, target_height + 1)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_height, new_width))

    return tf.keras.preprocessing.image.img_to_array(img)


def pad_image_height(img, target_height):
    a = tf.keras.preprocessing.image.img_to_array(img)

    height = a.shape[0]

    padding_amount = target_height - height

    assert padding_amount >= 0

    top_padding = padding_amount // 2
    if padding_amount % 2 == 0:
        vertical_padding = (top_padding, top_padding)
    else:
        vertical_padding = (top_padding, top_padding + 1)

    horizontal_padding = (0, 0)
    depth_padding = (0, 0)
    return scipy.pad(a, pad_width=[vertical_padding, horizontal_padding, depth_padding])


def pad_image(img, target_height, target_width):
    a = tf.keras.preprocessing.image.img_to_array(img)

    original_height, original_width, original_channels = a.shape

    im = np.ones((target_height, target_width, original_channels), dtype=np.float) * 255

    cropped = a[:target_height, :target_width]
    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            im[i, j, :] = a[i, j, :]

    return im


class BaseGenerator:
    def __iter__(self):
        raise NotImplementedError

    def get_image_array(self, image_path, target_height):
        #return get_zero_padded_array(image_path, target_height)
        return get_image_array(image_path, target_height)


class MyExampleGenerator(BaseGenerator):
    def __init__(self, dataset_root, image_height, batch_size=32):
        self._root = dataset_root
        self._image_height = image_height
        self._batch_size = batch_size

        self._label_words = []
        split_path, _ = os.path.split(dataset_root)
        with open(os.path.join(split_path, 'dictionary.txt')) as f:
            for line in f.readlines():
                self._label_words.append(line.rstrip('\n'))

        with open(os.path.join(self._root, 'paths_list.txt')) as f:
            self._paths = f.readlines()

    @property
    def num_classes(self):
        return len(self._label_words)

    @property
    def size(self):
        return len(self._paths)

    def __iter__(self):
        paths = self._paths

        while True:
            random.shuffle(paths)

            x_batch = []
            y_batch = []
            for path in paths:
                image_path = path.rstrip('\n')
                x, y = self.get_example(image_path)

                x_batch.append(x)
                y_batch.append(y)

                if len(x_batch) >= self._batch_size:
                    yield self.prepare_batch(x_batch, y_batch)

            if len(y_batch) > 0:
                yield self.prepare_batch(x_batch, y_batch)

    def _to_one_hot(self, label):
        y = np.zeros((1, self.num_classes))
        y[0, label] = 1.0
        return y

    def prepare_batch(self, x_batch, y_batch):
        a = x_batch[0]
        batch_size = len(y_batch)
        x_shape = (batch_size,) + a.shape[1:]

        batch = (np.array(x_batch).reshape(x_shape),
                 np.array(y_batch).reshape(batch_size, -1))
        x_batch[:] = []
        y_batch[:] = []
        return batch

    def get_example(self, image_path):
        im = self.get_image_array(image_path, self._image_height)

        x = im.reshape(1, *im.shape) / 255.0
        label = self.get_label(image_path)
        y = self._to_one_hot(label)
        return x, y

    def get_label(self, image_path):
        parent, file_name = os.path.split(image_path)
        _, label = os.path.split(parent)
        return int(label)


class CtcGenerator(MyExampleGenerator):
    def __init__(self, dataset_root, image_height, dictionary):
        super().__init__(dataset_root, image_height, batch_size=1)
        self._dictionary = dictionary

    def __iter__(self):
        paths = self._paths

        while True:
            random.shuffle(paths)

            for path in paths:
                image_path = path.rstrip('\n')
                image_array, labels = self.get_example(image_path)

                X = np.array(image_array).reshape(1, *image_array.shape)

                labels = np.array(labels, dtype=np.int32).reshape(1, len(labels))

                lstm_input_shape = compute_output_shape(image_array.shape)
                width, channels = lstm_input_shape
                input_lengths = np.array(width, dtype=np.int32).reshape(1, 1)
                label_lengths = np.array(len(labels[0]), dtype=np.int32).reshape(1, 1)

                yield [X, labels, input_lengths, label_lengths], labels

    def word_label_to_ascii_codes(self, label):
        word = self._dictionary[label]
        return [ord(ch) for ch in word]

    def get_example(self, image_path):
        im = self.get_image_array(image_path, self._image_height)

        x = im / 255.0
        label = self.get_label(image_path)
        y = self.word_label_to_ascii_codes(label)
        return x, y


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
    def __init__(self, dataset_root, char_table, image_height, batch_size=4, augment=False, batch_adapter=None):
        self._root = dataset_root
        self._char_table = char_table
        self._batch_size = batch_size
        self._augment = augment

        self._preprocessor = BasePreprocessor(dataset_root, image_height, augment)
        self._preprocessor.fit()

        if batch_adapter is None:
            self._adapter = CTCAdapter()
        else:
            self._adapter = batch_adapter

        self._ds = CompiledDataset(dataset_root)

        self._image_height = image_height

        self._indices = list(range(self._ds.size))

    @property
    def size(self):
        return self._ds.size

    @property
    def image_height(self):
        return self._image_height

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
        x = self._preprocessor.process(image_path)
        y = self.text_to_class_labels(text)
        return x, y


def prepare_x(image_path, image_height, should_binarize=True, transform=False):
    image_array = get_image_array(image_path, image_height)
    if should_binarize:
        a = binarize(image_array)
    else:
        a = image_array

    if transform:
        rotation_range = 1
        shift = 3
        zoom = 0.01
        image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=rotation_range, width_shift_range=shift, height_shift_range=shift,
            zoom_range=zoom
        )
        gen = image_gen.flow(np.array([a]), batch_size=1)

        a = next(gen)[0]

    return a / 255.0


def binarize(image_array, threshold=200, invert=True):
    h, w, channels = image_array.shape
    grayscale = rgb_to_grayscale(image_array)
    black_mask = grayscale < threshold
    white_mask = grayscale >= threshold

    if invert:
        tmp = white_mask
        white_mask = black_mask
        black_mask = tmp

    grayscale[white_mask] = 255
    grayscale[black_mask] = 0

    return grayscale.reshape((h, w, 1))


def rgb_to_grayscale(a):
    return a[:, :, 0] * 0.2125 + a[:, :, 1] * 0.7154 + a[:, :, 2] * 0.0721


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


class BasePreprocessor:
    def __init__(self, dataset_root, image_height, augment):
        self._image_height = image_height
        self._augment = augment

    def fit(self):
        pass

    def process(self, image_path):
        return prepare_x(image_path, self._image_height, transform=self._augment)


# todo: Create preprocessor class
