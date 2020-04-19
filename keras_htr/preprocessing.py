import numpy as np
import json
import tensorflow as tf

from keras_htr.generators import CompiledDataset
import scipy


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


def pad_array_width(a, target_width):
    width = a.shape[1]

    right_padding = target_width - width

    if right_padding < 0:
        # if image width is larger than target_width, crop the image
        return a[:, :target_width]

    horizontal_padding = (0, right_padding)
    vertical_padding = (0, 0)
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


class BasePreprocessor:
    def fit(self, train_path, val_path, test_path):
        pass

    def configure(self, **kwargs):
        pass

    def process(self, image_path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def _save_dict(self, path, d):
        s = json.dumps(d)
        with open(path, 'w') as f:
            f.write(s)


class Cnn1drnnCtcPreprocessor(BasePreprocessor):
    def __init__(self):
        self._average_height = 50

    def configure(self, average_height=50):
        self._average_height = average_height

    def fit(self, train_path, val_path, test_path):
        train_ds = CompiledDataset(train_path)
        self._average_height = train_ds.average_height

    def process(self, image_path):
        image_array = get_image_array(image_path, self._average_height)
        a = binarize(image_array)
        return tf.keras.preprocessing.image.array_to_img(a)

    def save(self, path):
        d = {'average_height': self._average_height}
        self._save_dict(path, d)


class EncoderDecoderPreprocessor(BasePreprocessor):
    def __init__(self):
        self._average_height = 50
        self._max_image_width = 0
        self._max_text_length = 0

    def configure(self, average_height=50, max_image_width=1200, max_text_length=100):
        self._average_height = average_height
        self._max_image_width = max_image_width
        self._max_text_length = max_text_length

    def fit(self, train_path, val_path, test_path):
        max_image_width_train, max_text_length_train = self.compute_ds_params(train_path)
        max_image_width_val, max_text_length_val = self.compute_ds_params(val_path)
        max_image_width_test, max_text_length_test = self.compute_ds_params(test_path)

        self._max_image_width = max(max_image_width_train, max_image_width_val, max_image_width_test)
        self._max_text_length = max(max_text_length_train, max_text_length_val, max_text_length_test)

    def compute_ds_params(self, ds_path):
        ds = CompiledDataset(ds_path)

        max_image_width = 0
        max_text_length = 0

        for image_path, text in ds:
            image_array = get_image_array(image_path, self._average_height)
            h, w, _ = image_array.shape
            max_image_width = max(max_image_width, w)
            max_text_length = max(max_text_length, len(text))

        return max_image_width, max_text_length

    def process(self, image_path):
        image_array = get_image_array(image_path, self._average_height)
        a = binarize(image_array)
        a = pad_array_width(a, self._max_image_width)
        return tf.keras.preprocessing.image.array_to_img(a)

    def save(self, path):
        d = dict(average_height=self._average_height,
                 max_image_width=self._max_image_width,
                 max_text_length=self._max_text_length)

        self._save_dict(path, d)
