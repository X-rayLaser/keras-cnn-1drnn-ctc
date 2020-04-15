import tensorflow as tf

from keras_htr import compute_output_shape
from keras_htr.models.base import create_conv_model, HTRModel


class ConvolutionalEncoderDecoderWithAttention(HTRModel):
    class InferenceModel:
        def __init__(self, encoder, decoder, attention, num_output_tokens):
            self._encoder = encoder
            self._decoder = decoder
            self._attention = attention
            self._num_output_tokens = num_output_tokens

        def predict(self, image_array, char_table):
            sos = char_table.sos
            eos = char_table.eos

            encoder_activations, state_h, state_c = self._encoder(image_array)
            concatenator = tf.keras.layers.Concatenate(axis=1)

            import numpy as np
            y_prev = np.zeros((1, self._num_output_tokens))
            y_prev[0, sos] = 1.0

            s = ''
            while True:
                inputs = [encoder_activations, state_h, state_c]
                context = self._attention(inputs)

                z = concatenator([context, y_prev])

                y_hat, state_h, state_c = self._decoder([z, state_h, state_c])
                pmf = y_hat[0]

                code = np.argmax(pmf)

                y_prev = np.zeros((1, self._num_output_tokens))
                y_prev[0, code] = 1.0

                if code == eos or len(s) > 20:
                    break

                if code == sos:
                    continue

                ch = char_table.get_character(code)

                s += ch

            return s

    def __init__(self, height, units, output_size, max_image_width, max_text_length):
        channels = 1
        Tx = max_text_length
        context_size = units
        decoder_input_size = context_size + output_size

        encoder_inputs = tf.keras.layers.Input(shape=(height, max_image_width, channels))
        decoder_inputs = tf.keras.layers.Input(shape=(Tx, output_size))
        concatenator = tf.keras.layers.Concatenate(axis=1)

        encoder = make_encoder_model(height, channels, units)
        decoder = make_step_decoder_model(units, decoder_input_size, output_size)

        num_activations, _ = compute_output_shape((height, max_image_width, 1))
        attention = make_attention_model(num_activations=num_activations, encoder_num_units=units)

        x = encoder_inputs
        encoder_activations, state_h, state_c = encoder(x)

        outputs = []
        for t in range(Tx):
            print('STEP', t)
            inputs = [encoder_activations, state_h, state_c]
            context = attention(inputs)
            y = tf.keras.layers.Lambda(lambda x: x[:, t, :])(decoder_inputs)

            z = concatenator([context, y])

            y_hat, state_h, state_c = decoder([z, state_h, state_c])
            outputs.append(y_hat)

        self.training_model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

        self._encoder = encoder
        self._decoder = decoder
        self._attention = attention
        self._num_output_tokens = output_size

    @property
    def inference_model(self):
        model = self.InferenceModel(self._encoder, self._decoder, self._attention, self._num_output_tokens)
        return model


def make_encoder_model(height, channels, units):
    conv_net = create_conv_model(height, channels)
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