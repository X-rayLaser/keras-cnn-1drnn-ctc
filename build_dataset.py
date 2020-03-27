from iam_ondb._utils import file_iterator
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError
import os
import numpy as np
from util import FileCopier, PathFinder, extract_line_tags


def extract_texts_with_ids(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ParseError:
        raise Exception()

    root_tag = list(root.iterfind('handwritten-part'))
    assert len(root_tag) != 0

    for line in root_tag[0].iterfind('line'):
        for word in line.iterfind('word'):
            assert 'text' in word.attrib and 'id' in word.attrib
            text = word.attrib['text']
            file_id = word.attrib['id']
            yield text, file_id


def build_words_dataset(words_root='iam_database/iam_words',
                        xml_root='iam_database/iam_database_xml',
                        destination_folder='words_dataset',
                        size=10000, train_fraction=0.6, val_fraction=0.2):
    if os.path.exists(destination_folder):
        raise Exception('Data set already exists!')

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    label_maker = LabelMaker()

    count = 0

    train_folder = os.path.join(destination_folder, 'train')
    val_folder = os.path.join(destination_folder, 'validation')
    test_folder = os.path.join(destination_folder, 'test')
    train_dataset_creator = DataSetCreator(words_root, train_folder, label_maker)
    val_dataset_creator = DataSetCreator(words_root, val_folder, label_maker)
    test_dataset_creator = DataSetCreator(words_root, test_folder, label_maker)

    creators = [train_dataset_creator, val_dataset_creator, test_dataset_creator]

    for word, file_id in get_words_with_file_ids(xml_root):
        if count > size:
            break
        print(word, file_id, count, size)
        test_fraction = 1 - train_fraction - val_fraction
        pmf = [train_fraction, val_fraction, test_fraction]
        dataset_creator = np.random.choice(creators, p=pmf)

        dataset_creator.add_example(word, file_id)
        count += 1

    for dataset_creator in creators:
        dataset_creator.create_paths_file()

    dictionary_file = os.path.join(destination_folder, 'dictionary.txt')

    with open(dictionary_file, 'w') as f:
        for word in label_maker.words:
            f.write(word + '\n')


def get_words_with_file_ids(xml_root):
    for xml_path in file_iterator(xml_root):
        for word, file_id in extract_texts_with_ids(xml_path):
            if word.isalnum():
                yield word, file_id


class DataSetCreator:
    def __init__(self, words_root, destination, label_maker):
        if not os.path.exists(destination):
            os.makedirs(destination)

        self._finder = PathFinder(words_root)
        self._label_to_copier = {}
        self._label_maker = label_maker
        self._destination_folder = destination
        self._dataset_paths = []

    def add_example(self, word, file_id):
        file_path = self._finder.find_path(file_id)

        self._label_maker.make_label_if_not_exists(word)
        label = self._label_maker[word]
        label_string = str(label)

        if label_string not in self._label_to_copier:
            folder_path = os.path.join(self._destination_folder, label_string)
            self._label_to_copier[label_string] = FileCopier(folder_path)

        copier = self._label_to_copier[label_string]
        copy_path = copier.copy_file(file_path)

        self._dataset_paths.append(copy_path)

    def create_paths_file(self):
        paths_file = os.path.join(self._destination_folder, 'paths_list.txt')

        with open(paths_file, 'w') as f:
            for path in self._dataset_paths:
                f.write(path + '\n')


class LabelMaker:
    def __init__(self):
        self._word_to_label = {}
        self._label_to_word = []

    @property
    def num_labels(self):
        return len(self._word_to_label)

    @property
    def words(self):
        return self._label_to_word

    def make_label_if_not_exists(self, word):
        if word not in self._word_to_label:
            self._label_to_word.append(word)
            self._word_to_label[word] = self.num_labels

    def __getitem__(self, word):
        return self._word_to_label[word]


if __name__ == '__main__':
    build_words_dataset(size=100, train_fraction=0.8, val_fraction=0.1)
