class CharTable:
    def __init__(self, char_table_path):
        self._char_to_label, self._label_to_char = self.load_char_table(char_table_path)

    def load_char_table(self, path):
        char_to_label = {}
        label_to_char = {}
        with open(path) as f:
            for label, line in enumerate(f.readlines()):
                ch = line.rstrip('\n')
                char_to_label[ch] = label
                label_to_char[label] = ch

        return char_to_label, label_to_char

    @property
    def num_labels(self):
        return len(self._char_to_label)

    def get_label(self, ch):
        return self._char_to_label[ch]

    def get_character(self, class_label):
        return self._label_to_char[class_label]
