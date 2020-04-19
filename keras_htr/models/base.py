import tensorflow as tf
import json
import os


class HTRModel:
    def get_preprocessor(self):
        raise NotImplementedError

    def get_adapter(self):
        raise NotImplementedError

    def fit(self, train_generator, val_generator, compilation_params=None,
            training_params=None, **kwargs):
        raise NotImplementedError

    def predict(self, inputs, **kwargs):
        raise NotImplementedError

    def save(self, path, preprocessing_params):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save_model_params(self, params_path, model_class_name, model_params, preprocessing_params):
        d = {
            'model_class_name': model_class_name,
            'params': model_params,
            'preprocessing': preprocessing_params
        }

        s = json.dumps(d)
        with open(params_path, 'w') as f:
            f.write(s)

    @staticmethod
    def create(model_path):
        from .cnn_1drnn_ctc import CtcModel
        from .encoder_decoder import ConvolutionalEncoderDecoderWithAttention

        params_path = os.path.join(model_path, 'params.json')
        with open(params_path) as f:
            s = f.read()
        d = json.loads(s)

        class_name = d['model_class_name']

        if class_name == 'CtcModel':
            model = CtcModel.load(model_path)
        else:
            model = ConvolutionalEncoderDecoderWithAttention.load(model_path)

        return model


def compute_output_shape(input_shape):
    height, width, channels = input_shape
    new_width = width // 2 // 2 // 2
    return new_width, 80


def create_conv_model(channels=3):
    def concat(X):
        t = tf.keras.layers.Concatenate(axis=1)(tf.unstack(X, axis=3))
        return tf.transpose(t, [0, 2, 1])

    column_wise_concat = tf.keras.layers.Lambda(concat)
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(input_shape=(None, None, channels), filters=16, kernel_size=(3, 3),
                                     padding='same', activation=None))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation=None))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), padding='same', activation=None))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(column_wise_concat)

    return model
