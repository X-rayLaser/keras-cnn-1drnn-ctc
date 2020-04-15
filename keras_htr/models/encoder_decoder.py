import tensorflow as tf

from keras_htr import compute_output_shape
from keras_htr.models.base import create_conv_model, HTRModel
import math
import os
import json


class ConvolutionalEncoderDecoderWithAttention(HTRModel):
    class InferenceModel:
        def __init__(self, encoder, decoder, attention, num_output_tokens):
            self._encoder = encoder
            self._decoder = decoder
            self._attention = attention
            self._num_output_tokens = num_output_tokens

        def predict(self, image_array, sos, eos):
            encoder_activations, state_h, state_c = self._encoder(image_array)
            concatenator = tf.keras.layers.Concatenate(axis=1)

            import numpy as np
            y_prev = np.zeros((1, self._num_output_tokens))
            y_prev[0, sos] = 1.0

            labels = []
            while True:
                inputs = [encoder_activations, state_h, state_c]
                context = self._attention(inputs)

                z = concatenator([context, y_prev])

                y_hat, state_h, state_c = self._decoder([z, state_h, state_c])
                pmf = y_hat[0]

                code = np.argmax(pmf)

                y_prev = np.zeros((1, self._num_output_tokens))
                y_prev[0, code] = 1.0

                if code == eos or len(labels) > 50:
                    break

                if code == sos:
                    continue

                labels.append(code)

            return labels

    def __init__(self, height, units, output_size, max_image_width, max_text_length):
        self._height = height
        self._units = units
        self._output_size = output_size
        self._max_image_width = max_image_width
        self._max_text_length = max_text_length

        channels = 1
        Tx = max_text_length
        context_size = units
        decoder_input_size = context_size + output_size

        encoder_inputs = tf.keras.layers.Input(shape=(height, max_image_width, channels))
        decoder_inputs = tf.keras.layers.Input(shape=(Tx, output_size))

        encoder = make_encoder_model(height, channels, units)
        decoder = make_step_decoder_model(units, decoder_input_size, output_size)

        num_activations, _ = compute_output_shape((height, max_image_width, 1))
        attention = make_attention_model(num_activations=num_activations, encoder_num_units=units)

        self._encoder_inputs = encoder_inputs
        self._decoder_inputs = decoder_inputs

        self._encoder = encoder
        self._decoder = decoder
        self._attention = attention
        self._num_output_tokens = output_size

    @property
    def inference_model(self):
        model = self.InferenceModel(self._encoder, self._decoder, self._attention, self._num_output_tokens)
        return model

    def _create_training_model(self):
        x = self._encoder_inputs
        encoder_activations, state_h, state_c = self._encoder(x)
        concatenator = tf.keras.layers.Concatenate(axis=1)

        outputs = []
        for t in range(self._max_text_length):
            print('STEP', t)
            inputs = [encoder_activations, state_h, state_c]
            context = self._attention(inputs)
            y = tf.keras.layers.Lambda(lambda x: x[:, t, :])(self._decoder_inputs)

            z = concatenator([context, y])

            y_hat, state_h, state_c = self._decoder([z, state_h, state_c])
            outputs.append(y_hat)

        return tf.keras.Model([self._encoder_inputs, self._decoder_inputs], outputs)

    def fit(self, train_generator, val_generator, *args, **kwargs):
        steps_per_epoch = math.ceil(train_generator.size / train_generator.batch_size)
        val_steps = math.ceil(val_generator.size / val_generator.batch_size)

        training_model = self._create_training_model()

        training_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=[])

        training_model.fit(train_generator.__iter__(), steps_per_epoch=steps_per_epoch,
                           validation_data=val_generator.__iter__(), validation_steps=val_steps, *args, **kwargs)

    def predict(self, image_array, **kwargs):
        model = self.InferenceModel(self._encoder, self._decoder, self._attention, self._num_output_tokens)
        char_table = kwargs['char_table']
        sos = char_table.sos
        eos = char_table.eos
        return model.predict(image_array, sos, eos)

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        params_path = os.path.join(path, 'params.json')
        encoder_weights_path = os.path.join(path, 'encoder.h5')
        decoder_weights_path = os.path.join(path, 'decoder.h5')
        attention_weights_path = os.path.join(path, 'attention.h5')

        self.save_model_params(params_path, 'ConvolutionalEncoderDecoderWithAttention',
                               height=self._height, units=self._units,
                               output_size=self._output_size,
                               max_image_width=self._max_image_width,
                               max_text_length=self._max_text_length)

        self._encoder.save(encoder_weights_path)
        self._decoder.save(decoder_weights_path)
        self._attention.save(attention_weights_path)

    @classmethod
    def load(cls, path):
        params_path = os.path.join(path, 'params.json')
        encoder_weights_path = os.path.join(path, 'encoder.h5')
        decoder_weights_path = os.path.join(path, 'decoder.h5')
        attention_weights_path = os.path.join(path, 'attention.h5')

        with open(params_path) as f:
            s = f.read()

        d = json.loads(s)

        params = d['params']
        instance = cls(**params)

        instance._encoder = tf.keras.models.load_model(encoder_weights_path)
        instance._decoder = tf.keras.models.load_model(decoder_weights_path)
        instance._attention = tf.keras.models.load_model(attention_weights_path)

        return instance


def make_encoder_model(height, channels, units):
    conv_net = create_conv_model(channels)
    encoder_inputs = tf.keras.layers.Input(shape=(height, None, channels))
    x = encoder_inputs
    features = conv_net(x)

    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(features)

    return tf.keras.Model(encoder_inputs, [encoder_outputs, state_h, state_c])


def make_step_decoder_model(units, input_size, output_size):
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_inputs = tf.keras.layers.Input(shape=(1, input_size))

    decoder_states = [tf.keras.layers.Input(shape=(units,)),
                      tf.keras.layers.Input(shape=(units,))]

    reshapor = tf.keras.layers.Reshape(target_shape=(1, input_size))

    flattener = tf.keras.layers.Reshape(target_shape=(output_size,))

    decoder_x = reshapor(decoder_inputs)

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_x,
                                                     initial_state=decoder_states)
    decoder_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_size, activation='softmax')
    )
    y_hat = decoder_dense(decoder_outputs)

    y_hat = flattener(y_hat)
    return tf.keras.Model([decoder_inputs] + decoder_states, [y_hat, state_h, state_c])


def make_attention_model(num_activations, encoder_num_units):
    repeater = tf.keras.layers.RepeatVector(num_activations)
    concatenator = tf.keras.layers.Concatenate(axis=2)

    densor1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=10, activation='relu'))
    densor2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='linear'))

    softmax = tf.keras.layers.Softmax(axis=1)

    dotter = tf.keras.layers.Dot(axes=(1, 1))
    reshapor = tf.keras.layers.Reshape(target_shape=(encoder_num_units,))

    state_h = tf.keras.layers.Input(shape=(encoder_num_units,))
    state_c = tf.keras.layers.Input(shape=(encoder_num_units,))
    encoder_states = [state_h, state_c]

    encoder_activations = tf.keras.layers.Input(shape=(num_activations, encoder_num_units))

    state_h = repeater(state_h)
    state_c = repeater(state_c)

    x = concatenator([state_h, state_c, encoder_activations])

    x = densor1(x)
    x = densor2(x)
    alphas = softmax(x)

    context = dotter([alphas, encoder_activations])
    context = reshapor(context)

    return tf.keras.Model([encoder_activations] + encoder_states, context)