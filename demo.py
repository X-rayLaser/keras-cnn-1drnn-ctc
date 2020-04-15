from keras_htr import CERevaluator, predict_labels, cer_on_batch
from keras_htr.generators import LinesGenerator
import tensorflow as tf
import numpy as np
from keras_htr.char_table import CharTable
import os
import json
from keras_htr.models.cnn_1drnn_ctc import CtcModel
from keras_htr.models.encoder_decoder import ConvolutionalEncoderDecoderWithAttention
from keras_htr.generators import ConvolutionalEncoderDecoderAdapter, CTCAdapter


def codes_to_string(codes, char_table):
    return ''.join([char_table.get_character(code) for code in codes])


def run_demo_for_ctc_model(gen):

    for x_batch, _ in gen.__iter__():
        (X, labels, input_lengths, label_lengths) = x_batch
        predictions = model.predict(X, input_lengths=input_lengths, char_table=char_table)
        cer = cer_on_batch(model, x_batch)
        print(predictions.shape)
        prediction = codes_to_string(predictions[0], char_table)
        ground_true = codes_to_string(labels[0], char_table)
        a = np.array(X[0] * 255)
        im = tf.keras.preprocessing.image.array_to_img(a)
        im.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true, prediction))
        input('Press any key to see next example')


def run_demo_for_attention_model(gen, char_table):
    for x_batch, decoder_y in gen.__iter__():
        X, decoder_x = x_batch
        expected = ''
        for v in decoder_x[0]:
            label = v.argmax()
            if label == char_table.sos or label == char_table.eos:
                continue
            expected += char_table.get_character(label)

        image_array = X[0]
        predictions = model.predict(X, char_table=char_table)

        #cer = cer_on_batch(model, x_batch)
        cer = 1
        prediction = codes_to_string(predictions, char_table)
        ground_true = expected
        a = image_array * 255
        im = tf.keras.preprocessing.image.array_to_img(a)
        im.show()
        print('LER {}, "{}" -> "{}"'.format(cer, ground_true, prediction))
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
    if class_name == 'CtcModel':
        lines_generator = LinesGenerator(dataset_path, char_table, batch_size=1, batch_adapter=CTCAdapter())
        model = CtcModel.load(model_path)
        run_demo_for_ctc_model(lines_generator)
    else:
        model = ConvolutionalEncoderDecoderWithAttention.load(model_path)
        lines_generator = LinesGenerator(
            dataset_path, char_table, batch_size=1,
            batch_adapter=ConvolutionalEncoderDecoderAdapter(char_table=char_table,
                                                             max_image_width=model._max_image_width,
                                                             max_text_length=model._max_text_length)
        )

        run_demo_for_attention_model(lines_generator, char_table)


# todo: do polymorphism
# todo: refactor
# todo: fix cer_on_batch
