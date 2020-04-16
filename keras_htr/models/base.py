import tensorflow as tf
import json


class HTRModel:
    def fit(self, train_gen, val_gen, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save_model_params(self, params_path, model_class_name, **params):
        d = {
            'model_class_name': model_class_name,
            'params': params
        }

        s = json.dumps(d)
        with open(params_path, 'w') as f:
            f.write(s)


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