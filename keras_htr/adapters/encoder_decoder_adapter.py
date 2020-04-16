import numpy as np
import tensorflow as tf

from keras_htr.adapters.base import BatchAdapter


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

        decoder_x = np.concatenate([sos_column, padded_labellings], axis=1)
        decoder_y = np.concatenate([padded_labellings, eos_column], axis=1)

        decoder_x = tf.keras.utils.to_categorical(decoder_x, num_classes=self._num_classes)
        decoder_y = tf.keras.utils.to_categorical(decoder_y, num_classes=self._num_classes)

        decoder_y = list(np.swapaxes(decoder_y, 0, 1))

        return [x, decoder_x], decoder_y