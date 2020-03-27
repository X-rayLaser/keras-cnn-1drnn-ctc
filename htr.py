import tensorflow as tf
import argparse
from toolkit import recognize_document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image

    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    batch_size, image_height, image_width, channels = model.input_shape

    res = recognize_document(model, image_path, image_height)
    print('Recognized text: "{}"'.format(res))
