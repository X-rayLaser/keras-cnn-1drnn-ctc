from toolkit import CERevaluator, get_meta_info, predict_labels, cer_on_batch
from toolkit.generators import LinesGenerator
import tensorflow as tf
import numpy as np


def codes_to_string(codes):
    return ''.join([chr(code) for code in codes])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    meta_info = get_meta_info()
    image_height = meta_info['average_height']

    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})

    lines_generator = LinesGenerator(dataset_path, image_height, batch_size=1)

    cer = CERevaluator(model, lines_generator, steps=None)

    for x_batch, _ in lines_generator.__iter__():
        (X, labels, input_lengths, label_lengths) = x_batch
        predictions = predict_labels(model, X, input_lengths)
        cer = cer_on_batch(model, x_batch)
        prediction = codes_to_string(predictions[0])
        ground_true = codes_to_string(labels[0])
        a = np.array(X[0] * 255)
        im = tf.keras.preprocessing.image.array_to_img(a)
        im.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true, prediction))
        input('Press any key to see next example')
