import os
from xml.etree.ElementTree import ParseError
from xml.etree import ElementTree as ET
from keras_htr.data_source.base import Source


class IAMSource(Source):
    def __init__(self, xml_root='iam_database/xml',
                 line_images_root='iam_database/lines'):
        self._xml_root = xml_root
        self._images_root = line_images_root
        self._validate_paths()

    def _validate_paths(self):
        if not (os.path.isdir(self._xml_root) and os.path.isdir(self._images_root)):
            iam_url = 'http://www.fki.inf.unibe.ch/databases/iam-handwriting-database'
            msg = "IAM database directories missing: {} and {}. " \
                  "If you don't have IAM database, you can " \
                  "download it from this URL {}".format(self._xml_root,
                                                        self._images_root,
                                                        iam_url)
            raise Exception(msg)

    def __iter__(self):
        finder = LineImageFinder(self._images_root)

        for line, file_path in get_lines_with_file_paths(self._xml_root, finder):
            yield file_path, line


def file_iterator(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            yield os.path.join(dirpath, name)


def extract_line_tags(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ParseError:
        raise Exception()

    root_tag = list(root.iterfind('handwritten-part'))
    assert len(root_tag) != 0

    for line in root_tag[0].iterfind('line'):
        yield line


class PathFinder:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def find_path(self, object_id):
        self.validate_id(object_id)
        dir_candidate = self._get_directory_path(self.root_dir, object_id)

        dir_path = self._choose_path(dir_candidate)
        last_part = self._last_id_part(object_id)

        for file_name in os.listdir(dir_path):
            name_without_extension, _ = os.path.splitext(file_name)
            if self._last_id_part(name_without_extension) == last_part:
                return os.path.join(dir_path, file_name)

        raise ObjectDoesNotExistError(object_id)

    def validate_id(self, object_id):
        validate_id(object_id)

    def _choose_path(self, dir_candidate):
        chomped_path = self._chomp_letter(dir_candidate)

        if os.path.isdir(dir_candidate):
            return dir_candidate
        elif os.path.isdir(chomped_path):
            return chomped_path
        else:
            raise ObjectDoesNotExistError()

    def _last_id_part(self, object_id):
        return '-'.join(object_id.split('-')[-2:])

    def _get_directory_path(self, root, object_id):
        self.validate_id(object_id)

        parts = object_id.split('-')
        folder = parts[0]
        subfolder = parts[0] + '-' + parts[1]
        return os.path.join(root, folder, subfolder)

    def _chomp_letter(self, s):
        if self._last_is_digit(s):
            return s
        return s[:-1]

    def _last_is_digit(self, s):
        return s[-1].isdigit()


def validate_id(id_string):
    parts = id_string.split('-')
    if len(parts) != 4:
        raise MalformedIdError(id_string)
    return validate_parts(parts)


def validate_parts(parts):
    for part in parts:
        if not part.isalnum():
            raise MalformedIdError(part)


class MalformedIdError(Exception):
    pass


class ObjectDoesNotExistError(Exception):
    pass


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


def get_lines_with_file_paths(xml_root, finder):
    for line, file_id in get_lines_with_file_ids(xml_root):
        file_path = finder.find_path(file_id)
        yield line, file_path
