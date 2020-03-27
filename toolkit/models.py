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

    new_width = ((width // 2 - 4) // 2 - 2) // 2
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

        #x = tf.keras.layers.Dropout(rate=0.5)(x)
        #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        #x = tf.keras.layers.Dropout(rate=0.5)(x)
        #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

        #x = tf.keras.layers.Dropout(rate=0.5)(x)
        #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)

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
