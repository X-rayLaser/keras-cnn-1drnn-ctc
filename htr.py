import tensorflow as tf
import argparse
from keras_htr.char_table import CharTable
from keras_htr.models.base import HTRModel
from keras_htr import codes_to_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('char_table', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--raw', type=bool, default=False)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    char_table_path = args.char_table
    raw = args.raw

    char_table = CharTable(char_table_path)

    model = HTRModel.create(model_path)
    adapter = model.get_adapter()

    if raw:
        preprocessor = model.get_preprocessor()
        image = preprocessor.process(image_path)
    else:
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")

    inputs = adapter.adapt_x(image)
    labels = model.predict(inputs)[0]
    res = codes_to_string(labels, char_table)

    print('Recognized text: "{}"'.format(res))
