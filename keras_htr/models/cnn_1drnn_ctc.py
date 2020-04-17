import tensorflow as tf

from .base import HTRModel, create_conv_model
import math
import os
import json


class CtcModel(HTRModel):
    def __init__(self, units, num_labels, height, channels=3):
        self._units = units
        self._num_labels = num_labels
        self._height = height
        self._channels = channels

        inp = tf.keras.layers.Input(shape=(height, None, channels))

        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
        densor = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_labels + 1, activation='softmax')
        )

        x = inp
        convnet = create_conv_model(channels)
        x = convnet(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = lstm(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        self.y_pred = densor(x)

        self.graph_input = inp

        self._weights_model = tf.keras.Model(self.graph_input, self.y_pred)
        self._preprocessing_options = {}

    def _create_training_model(self):
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args

            return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = tf.keras.layers.Input(name='the_labels',
                                       shape=[None], dtype='float32')
        input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

        loss_out = tf.keras.layers.Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([self.y_pred, labels, input_length, label_length])

        return tf.keras.Model(inputs=[self.graph_input, labels, input_length, label_length],
                              outputs=loss_out)

    def _create_inference_model(self):
        return tf.keras.Model(self.graph_input, self.y_pred)

    def fit(self, train_generator, val_generator, *args, **kwargs):
        steps_per_epoch = math.ceil(train_generator.size / train_generator.batch_size)
        val_steps = math.ceil(val_generator.size / val_generator.batch_size)

        loss = self._get_loss()
        lr = 0.001

        training_model = self._create_training_model()
        training_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=[])

        print(kwargs)
        training_model.fit(train_generator.__iter__(), steps_per_epoch=steps_per_epoch,
                           validation_data=val_generator.__iter__(), validation_steps=val_steps, *args, **kwargs)

    def _get_inference_model(self):
        return self._create_inference_model()

    def get_preprocessor(self):
        from keras_htr.preprocessing import Cnn1drnnCtcPreprocessor
        preprocessor = Cnn1drnnCtcPreprocessor()
        preprocessor.configure(**self._preprocessing_options)
        return preprocessor

    def get_adapter(self):
        from ..adapters.cnn_1drnn_ctc_adapter import CTCAdapter
        return CTCAdapter()

    def predict(self, inputs, **kwargs):
        X, input_lengths = inputs
        ypred = self._get_inference_model().predict(X)
        labels = decode_greedy(ypred, input_lengths)
        return labels

    def save(self, path, preprocessing_params):
        if not os.path.exists(path):
            os.mkdir(path)

        params_path = os.path.join(path, 'params.json')
        weights_path = os.path.join(path, 'weights.h5')

        model_params = dict(
            units=self._units,
            num_labels=self._num_labels,
            height=self._height,
            channels=self._channels
        )
        self.save_model_params(params_path, 'CtcModel', model_params, preprocessing_params)
        self._weights_model.save_weights(weights_path)

    @classmethod
    def load(cls, path):
        params_path = os.path.join(path, 'params.json')
        weights_path = os.path.join(path, 'weights.h5')
        with open(params_path) as f:
            s = f.read()

        d = json.loads(s)

        params = d['params']
        instance = cls(**params)

        instance._weights_model.load_weights(weights_path)
        instance._preprocessing_options = d['preprocessing']
        return instance

    def _get_loss(self):
        return {'ctc': lambda y_true, y_pred: y_pred}


def decode_greedy(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        inputs = tf.transpose(inputs, [1, 0, 2])
        decoded, _ = tf.nn.ctc_greedy_decoder(inputs, input_lengths.flatten())

        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res


def beam_search_decode(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        inputs = tf.transpose(inputs, [1, 0, 2])
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(inputs, input_lengths.flatten(), beam_width=10)
        print(log_probs)
        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res