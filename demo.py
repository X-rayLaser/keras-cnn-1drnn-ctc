from keras_htr.edit_distance import compute_cer
import tensorflow as tf
from keras_htr.char_table import CharTable
import os
import json
from keras_htr.models.cnn_1drnn_ctc import CtcModel
from keras_htr.models.encoder_decoder import ConvolutionalEncoderDecoderWithAttention
from keras_htr.generators import CompiledDataset
from keras_htr import codes_to_string


def run_demo(model, gen, char_table, adapter):
    for image_path, ground_true_text in gen.__iter__():
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")

        expected_labels = [[char_table.get_label(ch) for ch in ground_true_text]]

        inputs = adapter.adapt_x(image)

        predictions = model.predict(inputs)
        cer = compute_cer(expected_labels, predictions.tolist())[0]

        predicted_text = codes_to_string(predictions[0], char_table)

        image.show()
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

    ds = CompiledDataset(dataset_path)

    if class_name == 'CtcModel':
        model = CtcModel.load(model_path)
    else:
        model = ConvolutionalEncoderDecoderWithAttention.load(model_path)

    adapter = model.get_adapter()
    run_demo(model, ds, char_table, adapter=adapter)


# todo: do polymorphism
# todo: refactor
# todo: fix cer_on_batch
