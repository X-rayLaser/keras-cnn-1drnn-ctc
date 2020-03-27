from toolkit import CERevaluator
from toolkit.generators import LinesGenerator
import tensorflow as tf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--steps', type=int, default=200)

    args = parser.parse_args()
    model_path = args.model
    dataset_path = args.dataset

    steps = args.steps

    model = tf.keras.models.load_model(model_path)
    batch_size, image_height, image_width, channels = model.input_shape

    lines_generator = LinesGenerator(dataset_path, image_height, batch_size=1)

    evaluator = CERevaluator(model, lines_generator, steps=steps)

    cer = evaluator.evaluate()
    print('Average CER metric is {}'.format(cer))