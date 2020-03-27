import os
import logging
import math
from keras_htr import get_meta_info, CERevaluator
from keras_htr.generators import LinesGenerator
from keras_htr.models import CtcModel, decode_greedy
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


class DebugCallback(Callback):
    def __init__(self, train_gen, val_gen, ctc_model_factory, interval=10):
        super().__init__()
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._ctc_model_factory = ctc_model_factory
        self._interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._interval == 0 and epochs > 0:
            print('Predictions on training inputs:')
            self.show_predictions(self._train_gen)
            print('Predictions on validation inputs:')
            self.show_predictions(self._val_gen)

    def show_predictions(self, gen):
        for i, example in enumerate(gen.__iter__()):
            if i > 10:
                break

            (X, labels, input_lengths, label_lengths), labels = example

            expected = ''.join([chr(code) for code in labels[0]])

            ypred = self._ctc_model_factory.inference_model.predict(X)
            labels = decode_greedy(ypred, input_lengths)
            predicted = ''.join([chr(code) for code in labels[0]])

            print(expected, '->', predicted)


class CerCallback(Callback):
    def __init__(self, train_gen, val_gen, ctc_model_factory, steps=None, interval=10):
        super().__init__()
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._ctc_model_factory = ctc_model_factory
        self._steps = steps
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0:
            train_cer = self.compute_cer(self._train_gen)
            val_cer = self.compute_cer(self._val_gen)
            print('train CER {}; val CER {}'.format(train_cer, val_cer))

    def compute_cer(self, gen):
        cer = CERevaluator(self._ctc_model_factory.inference_model, gen, self._steps)
        return cer.evaluate()

    def show_output(self, X, input_lengths, labels):
        expected = ''.join([chr(code) for code in labels[0]])

        ypred = self._ctc_model_factory.inference_model.predict(X)
        labels = self._ctc_model_factory.decode_greedy(ypred, input_lengths)
        predicted = ''.join([chr(code) for code in labels[0]])
        print(expected, '->', predicted)


class CtcModelCheckpoint(Callback):
    def __init__(self, ctc_model, save_path):
        super().__init__()
        self._model = ctc_model
        self._save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self._model.inference_model.save(self._save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--debug_interval', type=int, default=10)
    parser.add_argument('--augment', type=bool, default=False)

    args = parser.parse_args()
    batch_size = args.batch_size
    units = args.units
    lr = args.lr
    epochs = args.epochs
    debug_interval = args.debug_interval
    augment = args.augment

    print('augment is {}'.format(augment))

    meta_info = get_meta_info()
    num_examples = meta_info['num_examples']
    image_height = meta_info['average_height']

    train_generator = LinesGenerator('lines_dataset/train', image_height, batch_size, augment=augment)
    val_generator = LinesGenerator('lines_dataset/validation', image_height, batch_size)

    ctc_model_factory = CtcModel(units=units, num_labels=128,
                                 height=train_generator.image_height, channels=1)
    model = ctc_model_factory.training_model
    loss = ctc_model_factory.get_loss()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=[])
    train_gen = train_generator.__iter__()
    val_gen = val_generator.__iter__()

    steps_per_epoch = math.ceil(train_generator.size / batch_size)
    val_steps = math.ceil(val_generator.size / batch_size)

    checkpoint = CtcModelCheckpoint(ctc_model_factory, 'conv_lstm_model.h5')

    train_debug_generator = LinesGenerator('lines_dataset/train', image_height, batch_size=1)
    val_debug_generator = LinesGenerator('lines_dataset/validation', image_height, batch_size=1)
    output_debugger = DebugCallback(train_debug_generator, val_debug_generator,
                                    ctc_model_factory, interval=debug_interval)

    cer_generator = LinesGenerator('lines_dataset/train', image_height, batch_size=1)
    cer_val_generator = LinesGenerator('lines_dataset/validation', image_height, batch_size=1)
    CER_metric = CerCallback(cer_generator, cer_val_generator,
                             ctc_model_factory, steps=16, interval=debug_interval)

    callbacks = [checkpoint, output_debugger, CER_metric]
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_data=val_gen, validation_steps=val_steps,
              callbacks=callbacks)
