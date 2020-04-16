from keras_htr import CERevaluator, predict_labels, cer_on_batch
from keras_htr.edit_distance import compute_cer
from keras_htr.generators import LinesGenerator
import tensorflow as tf
import numpy as np
from keras_htr.char_table import CharTable
import os
import json
from keras_htr.models.cnn_1drnn_ctc import CtcModel
from keras_htr.models.encoder_decoder import ConvolutionalEncoderDecoderWithAttention
from keras_htr.adapters.encoder_decoder_adapter import ConvolutionalEncoderDecoderAdapter
from keras_htr.adapters.cnn_1drnn_ctc_adapter import CTCAdapter


def codes_to_string(codes, char_table):
    return ''.join([char_table.get_character(code) for code in codes])


def run_demo_for_ctc_model(model, gen, char_table, adapter):
    for image_path, ground_true_text in gen.__iter__():
        img = tf.keras.preprocessing.image.load_img(image_path, grayscale=True)
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = [a / 255.0]
        expected_labels = [[char_table.get_label(ch) for ch in ground_true_text]]
        batch = (x, expected_labels)

        x_batch, y_batch = adapter.adapt_batch(batch)
        (X, labels, input_lengths, label_lengths) = x_batch

        predictions = model.predict(X, input_lengths=input_lengths, char_table=char_table)
        cer = cer_on_batch(model, x_batch)

        predicted_text = codes_to_string(predictions[0], char_table)

        img.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true_text, predicted_text))
        input('Press any key to see next example')


def run_demo_for_attention_model(model, gen, char_table, adapter):
    for image_path, ground_true_text in gen.__iter__():
        img = tf.keras.preprocessing.image.load_img(image_path, grayscale=True)
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = [a / 255.0]
        expected_labels = [[char_table.get_label(ch) for ch in ground_true_text]]
        batch = (x, expected_labels)

        x_batch, y_batch = adapter.adapt_batch(batch)
        X, decoder_x = x_batch

        predictions = model.predict(X, char_table=char_table)
        cer = compute_cer(expected_labels, predictions)

        predicted_text = codes_to_string(predictions[0], char_table)

        img.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true_text, predicted_text))
        input('Press any key to see next example')


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

    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        s = f.read()
    d = json.loads(s)

    class_name = d['model_class_name']
    from keras_htr.generators import CompiledDataset

    ds = CompiledDataset(dataset_path)

    if class_name == 'CtcModel':
        adapter = CTCAdapter()
        model = CtcModel.load(model_path)
        run_demo_for_ctc_model(model, ds, char_table, adapter=adapter)
    else:
        model = ConvolutionalEncoderDecoderWithAttention.load(model_path)

        adapter = ConvolutionalEncoderDecoderAdapter(char_table=char_table,
                                                     max_image_width=model._max_image_width,
                                                     max_text_length=model._max_text_length)

        run_demo_for_attention_model(model, ds, char_table, adapter=adapter)


# todo: do polymorphism
# todo: refactor
# todo: fix cer_on_batch
