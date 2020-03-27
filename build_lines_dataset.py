import os
from util import extract_line_tags, PathFinder, split_examples, FileCopier
from iam_ondb._utils import file_iterator, validate_parts
import json
import tensorflow as tf


class LineImageFinder(PathFinder):
    def validate_id(self, object_id):
        parts = object_id.split('-')
        if len(parts) != 3:
            raise Exception(object_id)
        return validate_parts(parts)

    def _last_id_part(self, object_id):
        return object_id.split('-')[-1]


def clean_line(line):
    line = line.replace('&quot;', '"')
    line = line.replace('&amp;', '&')
    return line


def extract_lines_with_ids(path):
    for line in extract_line_tags(path):
        assert 'text' in line.attrib and 'id' in line.attrib
        text = line.attrib['text']
        file_id = line.attrib['id']
        line = clean_line(text)
        yield line, file_id


def get_lines_with_file_ids(xml_root):
    for xml_path in file_iterator(xml_root):
        for line, file_id in extract_lines_with_ids(xml_path):
            yield line, file_id


def create_lines_dataset(xml_root='iam_database/iam_database_xml',
                         line_images_root='iam_database/iam_database_lines',
                         destination_folder='lines_dataset',
                         size=10000, train_fraction=0.6, val_fraction=0.2):
    finder = LineImageFinder(line_images_root)

    ids_generator = get_lines_with_file_ids(xml_root)

    dest_to_copier = {}
    dest_texts = {}

    num_created = 0
    for triple in split_examples(ids_generator, size, train_fraction, val_fraction):
        folder_name, text, file_id = triple
        split_destination = os.path.join(destination_folder, folder_name)
        if folder_name not in dest_to_copier:
            dest_to_copier[folder_name] = FileCopier(split_destination)

        if split_destination not in dest_texts:
            dest_texts[split_destination] = []

        copier = dest_to_copier[folder_name]
        file_path = finder.find_path(file_id)

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

        create_meta_information(split_folder)


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

    import numpy as np

    d = dict(max_width=int(np.max(widths)),
             max_height=int(np.max(heights)),
             min_width=int(np.min(widths)),
             min_height=int(np.min(heights)),
             average_width=int(np.mean(widths)),
             average_height=int(np.mean(heights)),
             num_examples=len(widths))

    s = json.dumps(d)
    meta_path = os.path.join(dataset_path, 'meta.json')
    with open(meta_path, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100)

    args = parser.parse_args()
    create_lines_dataset(size=args.size, train_fraction=0.8, val_fraction=0.1)
