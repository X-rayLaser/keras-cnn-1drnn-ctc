class BatchAdapter:
    def fit(self, batches):
        pass

    def adapt_x(self, image):
        raise NotImplementedError

    def adapt_batch(self, batch):
        raise NotImplementedError

    def _pad_labellings(self, labellings, target_length, padding_code=0):
        padded_labellings = []
        for labels in labellings:
            padding_size = target_length - len(labels)

            if padding_size < 0:
                # if labelling length is larger than target_length, chomp excessive characters off
                new_labelling = labels[:target_length]
            else:
                new_labelling = labels + [padding_code] * padding_size
            assert len(new_labelling) > 0
            padded_labellings.append(new_labelling)

        return padded_labellings

    def _pad_array_width(self, a, target_width):
        from keras_htr.preprocessing import pad_array_width
        return pad_array_width(a, target_width)

    def _pad_image_arrays(self, image_arrays, target_width):
        return [self._pad_array_width(a, target_width) for a in image_arrays]