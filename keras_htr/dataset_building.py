import os
import numpy as np
import shutil
import json
import tensorflow as tf


def create_lines_dataset(data_source,
                         preprocessor,
                         destination_folder='lines_dataset',
                         size=10000, train_fraction=0.6, val_fraction=0.2):
    temp_folder = os.path.join(destination_folder, 'extracted_lines')

    extract_lines(data_source, temp_folder, size, train_fraction, val_fraction)

    train_path = os.path.join(temp_folder, 'train')
    val_path = os.path.join(temp_folder, 'validation')
    test_path = os.path.join(temp_folder, 'test')

    preprocessor.fit(train_path, val_path, test_path)

    split_folders = ['train', 'validation', 'test']

    for folder in split_folders:
        src_dir = os.path.join(temp_folder, folder)
        dest_dir = os.path.join(destination_folder, folder)
        preprocess_images(src_dir, dest_dir, preprocessor)

    char_table_file_name = 'character_table.txt'
    char_table_src = os.path.join(temp_folder, char_table_file_name)
    char_table_dest = os.path.join(destination_folder, char_table_file_name)

    shutil.copyfile(char_table_src, char_table_dest)


def preprocess_images(source, destination, preprocessor):
    from keras_htr.generators import CompiledDataset

    shutil.copytree(source, destination)

    ds = CompiledDataset(destination)
    for image_path, _ in ds:
        img = preprocessor.process(image_path)
        img.save(image_path)

    meta_path = os.path.join(destination, 'meta.json')
    os.remove(meta_path)
    create_meta_information(destination)


def extract_lines(data_source,
                  destination_folder='lines_dataset',
                  size=10000, train_fraction=0.6, val_fraction=0.2):
    dest_to_copier = {}
    dest_texts = {}

    num_created = 0

    example_generator = data_source.__iter__()
    for triple in split_examples(example_generator, size, train_fraction, val_fraction):
        folder_name, file_path, text = triple
        split_destination = os.path.join(destination_folder, folder_name)
        if folder_name not in dest_to_copier:
            dest_to_copier[folder_name] = FileCopier(split_destination)

        if split_destination not in dest_texts:
            dest_texts[split_destination] = []

        copier = dest_to_copier[folder_name]

        copier.copy_file(file_path)

        dest_texts[split_destination].append(text)

        num_created += 1
        if num_created % 500 == 0:
            completed_percentage = num_created / float(size) * 100
            print('Created {} out of {} lines. {} % done'.format(
                num_created, size, completed_percentage)
            )

    for split_folder in dest_texts.keys():
        lines_path = os.path.join(split_folder, 'lines.txt')
        with open(lines_path, 'w') as f:
            for line in dest_texts[split_folder]:
                f.write(line + '\n')

        print('Creating meta information for {} split folder'.format(split_folder))
        create_meta_information(split_folder)

    print('Creating a character table')

    split_folders = dest_texts.keys()
    char_table_lines = create_char_table(split_folders)

    char_table_path = os.path.join(destination_folder, 'character_table.txt')
    with open(char_table_path, 'w') as f:
        f.write(char_table_lines)


class FileCopier:
    def __init__(self, folder):
        self._folder = folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        self._num_copied = len(os.listdir(self._folder))

    def copy_file(self, obj):
        if type(obj) is str:
            # obj must be path to image file
            file_path = obj
            _, ext = os.path.splitext(file_path)

            dest = os.path.join(self._folder, str(self._num_copied) + ext)
            shutil.copyfile(file_path, dest)
        else:
            # obj must be Pillow image
            dest = os.path.join(self._folder, str(self._num_copied) + '.png')
            obj.save(dest)

        self._num_copied += 1
        return dest


def split_examples(example_generator, size, train_fraction=0.6, val_fraction=0.2):
    train_folder = 'train'
    val_folder = 'validation'
    test_folder = 'test'

    folders = [train_folder, val_folder, test_folder]

    for count, example in enumerate(example_generator):
        if count > size:
            break

        test_fraction = 1 - train_fraction - val_fraction
        pmf = [train_fraction, val_fraction, test_fraction]
        destination = np.random.choice(folders, p=pmf)

        yield (destination,) + example


def create_meta_information(dataset_path):
    widths = []
    heights = []

    for fname in os.listdir(dataset_path):
        _, ext = os.path.splitext(fname)
        if ext != '.txt':
            image_path = os.path.join(dataset_path, fname)
            image = tf.keras.preprocessing.image.load_img(image_path)
            widths.append(image.width)
            heights.append(image.height)

    lines_path = os.path.join(dataset_path, 'lines.txt')

    text_lengths = []
    with open(lines_path) as f:
        for row in f.readlines():
            line = row.rstrip('\n')
            text_lengths.append(len(line))

    max_width = int(np.max(widths))
    max_height = int(np.max(heights))
    min_width = int(np.min(widths))
    min_height = int(np.min(heights))
    average_width = int(np.mean(widths))
    average_height = int(np.mean(heights))
    max_text_length = int(np.max(text_lengths))
    num_examples = len(widths)

    d = dict(max_width=max_width,
             max_height=max_height,
             min_width=min_width,
             min_height=min_height,
             average_width=average_width,
             average_height=average_height,
             max_text_length=max_text_length,
             num_examples=num_examples)

    s = json.dumps(d)
    meta_path = os.path.join(dataset_path, 'meta.json')
    with open(meta_path, 'w') as f:
        f.write(s)


def create_char_table(split_folders):
    chars = set()
    for folder in split_folders:
        lines_path = os.path.join(folder, 'lines.txt')
        with open(lines_path) as f:
            for line in f.readlines():
                text = line.rstrip()
                line_chars = list(text)
                chars = chars.union(line_chars)

    char_table = '\n'.join(list(chars))
    return char_table
