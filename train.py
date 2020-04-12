import os
import logging
import math
from keras_htr import get_meta_info, CERevaluator, decode_greedy
from keras_htr.generators import LinesGenerator, ConvolutionalEncoderDecoderAdapter
from keras_htr.models.encoder_decoder import ConvolutionalEncoderDecoderWithAttention
from keras_htr.models.cnn_1drnn_ctc import CtcModel
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from keras_htr.char_table import CharTable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


class DebugCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, ctc_model_factory, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._ctc_model_factory = ctc_model_factory
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            print('Predictions on training inputs:')
            self.show_predictions(self._train_gen)
            print('Predictions on validation inputs:')
            self.show_predictions(self._val_gen)

    def show_predictions(self, gen):
        for i, example in enumerate(gen.__iter__()):
            if i > 5:
                break

            (X, labels, input_lengths, label_lengths), labels = example
            expected = ''.join([self._char_table.get_character(code) for code in labels[0]])
            labels = self._ctc_model_factory.predict(X, input_lengths=input_lengths)

            #ypred = self._ctc_model_factory.inference_model.predict(X)
            #labels = decode_greedy(ypred, input_lengths)
            predicted = ''.join([self._char_table.get_character(code) for code in labels[0]])

            print(expected, '->', predicted)


class CerCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, ctc_model_factory, steps=None, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._ctc_model_factory = ctc_model_factory
        self._steps = steps
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            train_cer = self.compute_cer(self._train_gen)
            val_cer = self.compute_cer(self._val_gen)
            print('train CER {}; val CER {}'.format(train_cer, val_cer))

    def compute_cer(self, gen):
        cer = CERevaluator(self._ctc_model_factory.inference_model, gen, self._steps)
        return cer.evaluate()


class CtcModelCheckpoint(Callback):
    def __init__(self, ctc_model, save_path):
        super().__init__()
        self._model = ctc_model
        self._save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self._model.save(self._save_path)


class DebugAttentionModelCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, attention_model_factory, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._attention_model_factory = attention_model_factory
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            print('Predictions on training inputs:')
            self.show_predictions(self._train_gen)
            print('Predictions on validation inputs:')
            self.show_predictions(self._val_gen)

    def show_predictions(self, gen):
        for i, example in enumerate(gen.__iter__()):
            if i > 5:
                break

            [X, decoder_x], decoder_y = example

            import numpy as np

            expected = ''
            for v in decoder_x[0]:
                label = v.argmax()
                if label == self._char_table.sos or label == self._char_table.eos:
                    continue
                expected += self._char_table.get_character(label)

            predicted = self._attention_model_factory.inference_model.predict(X, self._char_table)
            print(expected, '->', predicted)


def fit_ctc_model(args):
    dataset_path = args.ds
    model_save_path = args.model_path
    batch_size = args.batch_size
    units = args.units
    lr = args.lr
    epochs = args.epochs
    debug_interval = args.debug_interval
    augment = args.augment

    print('augment is {}'.format(augment))

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'validation')

    meta_info = get_meta_info(path=train_path)
    num_examples = meta_info['num_examples']
    image_height = meta_info['average_height']

    char_table_path = os.path.join(dataset_path, 'character_table.txt')

    char_table = CharTable(char_table_path)

    train_generator = LinesGenerator(train_path, char_table, batch_size, augment=augment)
    val_generator = LinesGenerator(val_path, char_table, batch_size)

    model = CtcModel(units=units, num_labels=char_table.size,
                     height=image_height, channels=1)

    checkpoint = CtcModelCheckpoint(model, model_save_path)

    train_debug_generator = LinesGenerator(train_path, char_table, batch_size=1)
    val_debug_generator = LinesGenerator(val_path, char_table, batch_size=1)
    output_debugger = DebugCallback(char_table, train_debug_generator, val_debug_generator,
                                    model, interval=debug_interval)

    cer_generator = LinesGenerator(train_path, char_table, batch_size=1)
    cer_val_generator = LinesGenerator(val_path, char_table, batch_size=1)
    CER_metric = CerCallback(char_table, cer_generator, cer_val_generator,
                             model, steps=16, interval=debug_interval)

    callbacks = [checkpoint, output_debugger, CER_metric]

    model.fit(train_generator, val_generator, epochs=epochs, callbacks=callbacks)


def fit_attention_model(args):
    dataset_path = args.ds
    model_save_path = args.model_path
    batch_size = args.batch_size
    units = args.units
    lr = args.lr
    epochs = args.epochs
    debug_interval = args.debug_interval
    augment = args.augment

    print('augment is {}'.format(augment))

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'validation')
    test_path = os.path.join(dataset_path, 'test')

    meta_info = get_meta_info(path=train_path)
    image_height = meta_info['average_height']

    char_table_path = os.path.join(dataset_path, 'character_table.txt')

    char_table = CharTable(char_table_path)

    max_image_width = meta_info['max_width']
    max_text_length = max(get_meta_info(path=train_path)['max_text_length'], get_meta_info(val_path)['max_text_length'],
                          get_meta_info(test_path)['max_text_length'])

    adapter = ConvolutionalEncoderDecoderAdapter(char_table, max_image_width, max_text_length)

    train_generator = LinesGenerator(train_path, char_table, batch_size,
                                     augment=augment, batch_adapter=adapter)

    val_generator = LinesGenerator(val_path, char_table, batch_size,
                                   batch_adapter=adapter)

    model_factory = ConvolutionalEncoderDecoderWithAttention(height=image_height,
                                                             units=units, output_size=char_table.size,
                                                             max_image_width=max_image_width,
                                                             max_text_length=max_text_length)
    model = model_factory.training_model

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=[])
    train_gen = train_generator.__iter__()
    val_gen = val_generator.__iter__()

    steps_per_epoch = math.ceil(train_generator.size / batch_size)
    val_steps = math.ceil(val_generator.size / batch_size)

    train_debug_generator = LinesGenerator(train_path, char_table, batch_size=1,
                                           augment=augment, batch_adapter=adapter)
    val_debug_generator = LinesGenerator(val_path, char_table, batch_size=1,
                                         batch_adapter=adapter)
    output_debugger = DebugAttentionModelCallback(char_table, train_debug_generator, val_debug_generator,
                                                  model_factory, interval=debug_interval)

    callbacks = [output_debugger]
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_data=val_gen, validation_steps=val_steps, callbacks=callbacks)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str)
    parser.add_argument('--model_path', type=str, default='conv_lstm_model.h5')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--debug_interval', type=int, default=10)
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--arch', type=str, default='cnn-1drnn-ctc')

    args = parser.parse_args()

    if args.arch == 'cnn-1drnn-ctc':
        fit_ctc_model(args)
    elif args.arch == 'encoder-decoder-attention':
        fit_attention_model(args)
    else:
        raise Exception('{} model architecture is unrecognized'.format(args.arch))
