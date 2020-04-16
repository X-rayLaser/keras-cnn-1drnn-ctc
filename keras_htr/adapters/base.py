import scipy


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