import tensorflow as tf

from .base import HTRModel, create_conv_model
import math


class CtcModel(HTRModel):
    def __init__(self, units, num_labels, height, channels=3):
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

        self.training_model = tf.keras.Model(inputs=[self.graph_input, labels, input_length, label_length],
                                             outputs=loss_out)
        self._inference_model = tf.keras.Model(self.graph_input, self.y_pred)

    def fit(self, train_generator, val_generator, *args, **kwargs):
        steps_per_epoch = math.ceil(train_generator.size / train_generator.batch_size)
        val_steps = math.ceil(val_generator.size / val_generator.batch_size)

        loss = self._get_loss()
        lr = 0.001
        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=[])

        self.training_model.fit(train_generator.__iter__(), steps_per_epoch=steps_per_epoch,
                                validation_data=val_generator.__iter__(), validation_steps=val_steps, *args, **kwargs)

    def _get_inference_model(self):
        return self._inference_model
        #return tf.keras.Model(self.graph_input, self.y_pred)

    def predict(self, image_array, **kwargs):
        ypred = self._get_inference_model().predict(image_array)
        input_lengths = kwargs['input_lengths']
        labels = decode_greedy(ypred, input_lengths)
        return labels

    def save(self, path):
        self._get_inference_model().save(path)

    def load(self, path):
        inference_model = tf.keras.models.load_model(path)
        raise Exception('What to do now...')

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