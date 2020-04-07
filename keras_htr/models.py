import tensorflow as tf


def create_conv_rnn_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(10, 1), padding='valid', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Reshape((-1, 64)))
    model.add(tf.keras.layers.LSTM(units=512))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    return model


def compute_output_shape(input_shape):
    height, width, channels = input_shape

    #new_width = ((width // 2 - 4) // 2 - 2) // 2
    new_width = width // 2 // 2 // 2
    return new_width, 80


def create_conv_model(input_height=50, channels=3):
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


class CtcModel:
    def __init__(self, units, num_labels, height, channels=3):
        inp = tf.keras.layers.Input(shape=(height, None, channels))

        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
        densor = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_labels + 1, activation='softmax')
        )

        x = inp
        convnet = create_conv_model(height, channels)
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

    @property
    def inference_model(self):
        return tf.keras.Model(self.graph_input, self.y_pred)

    def compile_model(self, lrate):

        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(lrate), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'])
        self.training_model.summary()
        return self.training_model

    def get_loss(self):
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


class ConvolutionalEncoderDecoderWithAttention:
    def __init__(self, height, units=128, output_size=128, max_image_width=2000, max_text_length=200):
        self.training_model = conv_encoder_decoder_model_with_attention(height, units, output_size, max_image_width, max_text_length)

    @property
    def inference_model(self):
        return None


def conv_encoder_decoder_model_with_attention(height, units=128, output_size=128, max_image_width=2000, max_text_length=200):
    channels = 1
    Tx = max_text_length
    encoder_inputs = tf.keras.layers.Input(shape=(height, max_image_width, channels))
    decoder_inputs = tf.keras.layers.Input(shape=(Tx, output_size))
    concatenator = tf.keras.layers.Concatenate(axis=1)

    encoder = make_encoder_model(height, channels, units)
    decoder = make_step_decoder_model(units, output_size)

    num_activations, _ = compute_output_shape((height, max_image_width, 1))
    attention = make_attention_model(num_activations=num_activations, max_text_length=Tx, encoder_num_units=units)

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

    concatenator = tf.keras.layers.Concatenate(axis=0)
    reshapor = tf.keras.layers.Reshape(target_shape=(Tx, output_size))

    output_tensor = concatenator(outputs)
    output_tensor = reshapor(output_tensor)
    # todo: fix error
    return tf.keras.Model([encoder_inputs, decoder_inputs], output_tensor)


def make_encoder_model(height, channels, units):
    conv_net = create_conv_model(height, channels)
    encoder_inputs = tf.keras.layers.Input(shape=(height, None, channels))
    x = encoder_inputs
    features = conv_net(x)

    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(features)

    return tf.keras.Model(encoder_inputs, [encoder_outputs, state_h, state_c])


def make_step_decoder_model(units=128, output_size=128):
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_inputs = tf.keras.layers.Input(shape=(1, output_size))
    decoder_states = [tf.keras.layers.Input(shape=(units,)),
                      tf.keras.layers.Input(shape=(units,))]

    reshapor = tf.keras.layers.Reshape(target_shape=(1, output_size))

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


def make_attention_model(num_activations, max_text_length, encoder_num_units):
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
