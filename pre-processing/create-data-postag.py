import numpy as np
from collections import defaultdict
import random

matrix_file = open("train_data/postag_output_matrix_number.txt")
label_file = open("train_data/postag_output_label_number.txt")

label_dict = defaultdict()

classes = ['R', 'N', 'CH', 'V', 'C', 'M', 'A', 'E', 'L', 'P', 'Nc', 'Cc', 'Ny', 'Nb', 'I', 'Np', 'T', 'Z', 'X', 'Nu',
           'Y', 'Ab', 'Vy', 'Vb', 'B', 'Ni']


class Element:
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label

    def get_vector(self):
        return self.vector

    def get_label(self):
        return self.label


class Elements:
    def __init__(self):
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def get_element(self, index):
        return self.elements[index]

    def size(self):
        return len(self.elements)


def to_label(number_label):
    return classes[int(number_label)]


def add_elements(dicts, elements):
    l = elements.get_element(0).get_label()
    if l not in dicts.keys():
        dicts[l] = []
    dicts[l].append(elements)


in_long_word = False
current_long_word = Elements()
count = 0

for line in matrix_file:
    count += 1
    line = line.strip()
    label = label_file.readline().strip()

    if line is "" or label is "" or line is None or label is None:
        break

    if label not in label_dict.keys():
        label_dict[label] = []

    element = Element(line, label)
    elements = Elements()
    elements.add_element(element)
    add_elements(label_dict, elements)

train_distribution = {
    '0': 2000,
    '1': 2000,
    '2': 2000,
    '3': 2000,
    '4': 2000,
    '5': 2000,
    '6': 2000,
    '7': 2000,
    '8': 2000,
    '9': 2000,
    '10': 2000,
}

train_ratio = 0.8

train_mat_file = open("train_data/postag_full_train_matrix_file.txt", "w")
test_mat_file = open("train_data/postag_full_test_matrix_file.txt", "w")

train_label_file = open("train_data/postag_full_train_label_file.txt", "w")
test_label_file = open("train_data/postag_full_test_label_file.txt", "w")

train_label_file_string = open("train_data/postag_full_train_label_file_string.txt", "w")
test_label_file_string = open("train_data/postag_full_test_label_file_string.txt", "w")

for key in [str(x) for x in range(0, 26)]:
    # num = train_distribution[key]

    # if num > len(label_dict[key]):
    num = len(label_dict[key])

    train_num = int(num * train_ratio)
    test_num = int(num - train_num)

    train_elements_index = random.sample(range(0, len(label_dict[key])), train_num)

    # create index and remove training index
    test_elements_index = list(set(range(0, len(label_dict[key]))) - set(train_elements_index))

    for i in train_elements_index:
        for j in range(0, label_dict[key][i].size()):
            train_mat_file.write(label_dict[key][i].get_element(j).get_vector())
            train_mat_file.write('\n')

            train_label_file.write(label_dict[key][i].get_element(j).get_label())
            train_label_file.write('\n')

            train_label_file_string.write(classes[int(label_dict[key][i].get_element(j).get_label())])
            train_label_file_string.write('\n')

    for i in test_elements_index:
        for j in range(0, label_dict[key][i].size()):
            test_mat_file.write(label_dict[key][i].get_element(j).get_vector())
            test_mat_file.write('\n')

            test_label_file.write(label_dict[key][i].get_element(j).get_label())
            test_label_file.write('\n')

            test_label_file_string.write(classes[int(label_dict[key][i].get_element(j).get_label())])
            test_label_file_string.write('\n')

train_mat_file.close()
test_mat_file.close()
train_label_file.close()
test_label_file.close()
