import numpy as np

from keras_htr import compute_output_shape
from keras_htr.adapters.base import BatchAdapter


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