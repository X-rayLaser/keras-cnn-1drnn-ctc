import os
import logging
from keras_htr import get_meta_info, LEREvaluator, decode_greedy
from keras_htr.generators import LinesGenerator
from keras_htr.models.encoder_decoder import ConvolutionalEncoderDecoderWithAttention
from keras_htr.models.cnn_1drnn_ctc import CtcModel
from tensorflow.keras.callbacks import Callback
from keras_htr.char_table import CharTable
from keras_htr.generators import CompiledDataset
import tensorflow as tf
from keras_htr.edit_distance import compute_cer
from keras_htr import codes_to_string
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


class CerCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, model, steps=None, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._model = model
        self._steps = steps
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            train_cer = self.compute_ler(self._train_gen)
            val_cer = self.compute_ler(self._val_gen)
            print('train LER {}; val LER {}'.format(train_cer, val_cer))

    def compute_ler(self, gen):
        cer = LEREvaluator(self._model, gen, self._steps, self._char_table)
        return cer.evaluate()


class MyModelCheckpoint(Callback):
    def __init__(self, model, save_path, preprocessing_params):
        super().__init__()
        self._model = model
        self._save_path = save_path
        self._preprocessing_params = preprocessing_params

    def on_epoch_end(self, epoch, logs=None):
        self._model.save(self._save_path, self._preprocessing_params)


class DebugModelCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, attention_model, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._model = attention_model
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            print('Predictions on training inputs:')
            self.show_predictions(self._train_gen)
            print('Predictions on validation inputs:')
            self.show_predictions(self._val_gen)

    def show_predictions(self, gen):
        adapter = self._model.get_adapter()
        for i, example in enumerate(gen.__iter__()):
            image_path, ground_true_text = example
            if i > 5:
                break

            image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")

            expected_labels = [[self._char_table.get_label(ch) for ch in ground_true_text]]

            inputs = adapter.adapt_x(image)

            predictions = self._model.predict(inputs)
            cer = compute_cer(expected_labels, predictions.tolist())[0]

            predicted_text = codes_to_string(predictions[0], self._char_table)

            print('LER {}, "{}" -> "{}"'.format(cer, ground_true_text, predicted_text))


def fit_model(model, train_path, val_path, char_table, batch_size,
              debug_interval, model_save_path, epochs, augment, lr):
    path = Path(train_path)

    with open(os.path.join(path.parent, 'preprocessing.json')) as f:
        s = f.read()

    preprocessing_params = json.loads(s)

    adapter = model.get_adapter()

    train_generator = LinesGenerator(train_path, char_table, batch_size,
                                     augment=augment, batch_adapter=adapter)

    val_generator = LinesGenerator(val_path, char_table, batch_size,
                                   batch_adapter=adapter)

    train_debug_generator = CompiledDataset(train_path)
    val_debug_generator = CompiledDataset(val_path)
    output_debugger = DebugModelCallback(char_table, train_debug_generator, val_debug_generator,
                                         model, interval=debug_interval)

    checkpoint = MyModelCheckpoint(model, model_save_path, preprocessing_params)

    cer_generator = CompiledDataset(train_path)
    cer_val_generator = CompiledDataset(val_path)
    CER_metric = CerCallback(char_table, cer_generator, cer_val_generator,
                             model, steps=5, interval=debug_interval)

    callbacks = [checkpoint, output_debugger, CER_metric]

    compilation_params = dict(optimizer=tf.keras.optimizers.Adam(lr=lr))
    training_params = dict(epochs=epochs, callbacks=callbacks)
    model.fit(train_generator, val_generator, compilation_params, training_params)


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

    image_height = meta_info['average_height']

    char_table_path = os.path.join(dataset_path, 'character_table.txt')

    char_table = CharTable(char_table_path)

    model = CtcModel(units=units, num_labels=char_table.size,
                     height=image_height, channels=1)

    fit_model(model, train_path, val_path, char_table, batch_size, debug_interval, model_save_path, epochs, augment, lr)


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

    model = ConvolutionalEncoderDecoderWithAttention(height=image_height,
                                                     units=units, output_size=char_table.size,
                                                     max_image_width=max_image_width,
                                                     max_text_length=max_text_length + 1,
                                                     sos=char_table.sos, eos=char_table.eos)

    fit_model(model, train_path, val_path, char_table, batch_size, debug_interval, model_save_path, epochs, augment, lr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str)
    parser.add_argument('--model_path', type=str, default='conv_lstm_model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--debug_interval', type=int, default=10)
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--arch', type=str, default='cnn-1drnn-ctc')

    args = parser.parse_args()

    if args.arch == 'cnn-1drnn-ctc':
        fit_ctc_model(args)
    elif args.arch == 'cnn-encoder-decoder':
        fit_attention_model(args)
    else:
        raise Exception('{} model architecture is unrecognized'.format(args.arch))
