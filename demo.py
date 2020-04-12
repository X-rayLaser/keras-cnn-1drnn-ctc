from keras_htr import CERevaluator, predict_labels, cer_on_batch
from keras_htr.generators import LinesGenerator
import tensorflow as tf
import numpy as np
from keras_htr.char_table import CharTable
import os


def codes_to_string(codes, char_table):
    return ''.join([char_table.get_character(code) for code in codes])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset

    char_table_path = os.path.join(os.path.dirname(dataset_path), 'character_table.txt')
    char_table = CharTable(char_table_path)

    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    batch_size, image_height, image_width, channels = model.input_shape

    lines_generator = LinesGenerator(dataset_path, char_table, batch_size=1)

    cer = CERevaluator(model, lines_generator, steps=None)

    for x_batch, _ in lines_generator.__iter__():
        (X, labels, input_lengths, label_lengths) = x_batch
        predictions = predict_labels(model, X, input_lengths)
        cer = cer_on_batch(model, x_batch)
        prediction = codes_to_string(predictions[0], char_table)
        ground_true = codes_to_string(labels[0], char_table)
        a = np.array(X[0] * 255)
        im = tf.keras.preprocessing.image.array_to_img(a)
        im.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true, prediction))
        input('Press any key to see next example')
