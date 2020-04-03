from keras_htr.dataset_building import create_lines_dataset
import json
import os
from importlib import import_module
import shutil


def get_class(fully_qualified_name):
    parts = fully_qualified_name.split('.')
    module_path = '.'.join(parts[:-1])

    source_module = import_module(module_path)
    class_name = parts[-1]
    source_class = getattr(source_module, class_name)
    return source_class


def get_source(config):
    if os.path.isfile(config):
        with open(config) as f:
            s = f.read()

        d = json.loads(s)

        source_class_name = d['source_class']
        params = d['source_args']
    else:
        print('HERE')
        # treating config as fully-qualified class name
        source_class_name = config
        params = dict()

    try:
        source_class = get_class(source_class_name)
    except (ModuleNotFoundError, AttributeError):
        raise Exception('Failed importing class {}'.format(source_class_name))

    return source_class(**params)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='keras_htr.data_source.synthetic.SyntheticSource')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--destination', type=str, default='lines_dataset')

    args = parser.parse_args()

    source = get_source(args.config)
    destination = args.destination

    if not os.path.isdir(destination):
        os.makedirs(destination)

    response = input('All existing data in the directory will be erased. Continue (Y/N) ?')
    if response == 'Y':
        shutil.rmtree(destination)
        create_lines_dataset(source, destination_folder=destination,
                             size=args.size, train_fraction=0.8, val_fraction=0.1)
    else:
        print('Aborting...')
