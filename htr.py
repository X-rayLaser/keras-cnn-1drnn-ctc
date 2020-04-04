import tensorflow as tf
import argparse
from keras_htr import recognize_document
from keras_htr.char_table import CharTable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('char_table', type=str)
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    char_table_path = args.char_table

    char_table = CharTable(char_table_path)

    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    batch_size, image_height, image_width, channels = model.input_shape

    res = recognize_document(model, image_path, image_height, char_table)
    print('Recognized text: "{}"'.format(res))
