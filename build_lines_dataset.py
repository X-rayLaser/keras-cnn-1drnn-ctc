from keras_htr.dataset_building import create_lines_dataset
import json
import os
from importlib import import_module
import shutil
from keras_htr.data_source.synthetic import SyntheticSource
from keras_htr.data_source.iam import IAMSource


def get_class(fully_qualified_name):
    print(fully_qualified_name)
    parts = fully_qualified_name.split('.')
    module_path = '.'.join(parts[:-1])

    class_module = import_module(module_path)
    class_name = parts[-1]
    return getattr(class_module, class_name)


def get_source_class(class_name):
    # match short pseudo names for source
    if class_name == 'synthetic':
        return SyntheticSource

    if class_name == 'iam':
        return IAMSource

    return get_class(class_name)


def get_source(source_class_name):
    params = dict()

    try:
        source_class = get_source_class(source_class_name)
    except (ModuleNotFoundError, AttributeError):
        raise Exception('Failed importing class {}'.format(source_class_name))

    return source_class(**params)


def get_preprocessor(arch):
    allowed = ['cnn-1drnn-ctc', 'cnn-encoder-decoder']
    if arch == 'cnn-1drnn-ctc':
        fully_qualified_preprocessor = 'keras_htr.preprocessing.Cnn1drnnCtcPreprocessor'
    elif arch == 'cnn-encoder-decoder':
        fully_qualified_preprocessor = 'keras_htr.preprocessing.EncoderDecoderPreprocessor'
    else:
        s = ', '.join(allowed)
        raise Exception('"{}" model architecture is unrecognized. Valid options: {}'.format(args.arch, s))

    preprocessor_class = get_class(fully_qualified_preprocessor)
    return preprocessor_class()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--source', type=str, default='synthetic')
    parser.add_argument('--arch', type=str, default='cnn-1drnn-ctc')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--destination', type=str, default='lines_dataset')

    args = parser.parse_args()
    fully_qualified_source = args.source
    config = args.config
    arch = args.arch
    destination = args.destination
    size = args.size

    if config != '':
        with open(config) as f:
            s = f.read()

        d = json.loads(s)

        source_class_name = d['source_class']
        params = d['source_args']
        if 'arch' in d:
            arch = d['arch']

        source_class = get_source_class(source_class_name)
        source = source_class(**params)
    else:
        source = get_source(fully_qualified_source)

    preprocessor = get_preprocessor(arch)

    if not os.path.isdir(destination):
        os.makedirs(destination)

    response = input('All existing data in the directory {} '
                     'will be erased. Continue (Y/N) ?'.format(destination))
    if response == 'Y':
        shutil.rmtree(destination)
        create_lines_dataset(source, preprocessor, destination_folder=destination,
                             size=size, train_fraction=0.8, val_fraction=0.1)
    else:
        print('Aborting...')
