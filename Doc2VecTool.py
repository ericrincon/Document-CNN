__author__ = 'ericrincon'

import os
import numpy
import linecache
import multiprocessing

from random import shuffle
from gensim import utils

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from nltk.tokenize import sent_tokenize


class LabeledLineDocument(object):
    def __init__(self, source, shuffle_lines=True):


        self.n_lines = 0
        self.source = source
        self.n_list = []

        with utils.smart_open(self.source) as file:
            for line_number, line in enumerate(file):
                self.n_lines += 1
                self.n_list.append(line_number)
        if shuffle_lines:
            shuffle(self.n_list)

    def __iter__(self):

        for line_number in self.n_list:
            line = linecache.getline(self.source, line_number)
            yield TaggedDocument(utils.to_unicode(line).split(), [line_number])


def infer_matrix(x, model, model_vector_size):
    infered_matrix = numpy.zeros((x.shape[0], model_vector_size))
    for i in range(x.shape[0]):
        string = x[i, 0]

        for j in range(1, x.shape[1]):
            string += " " + x[i, j]
        preprocessed_line = preprocess_line(string)
        infered_matrix[i, :] = model.infer_vector(preprocessed_line)
    return infered_matrix


def create_dm_dbow_matrix(x, dm_model, dbow_model, model_vector_size, other_model_vector_size=None):

    if other_model_vector_size is None:
        other_model_vector_size = model_vector_size

    dm_matrix = infer_matrix(x, dm_model, model_vector_size)
    dbow_matrix = infer_matrix(x, dbow_model, other_model_vector_size)

    return numpy.hstack((dm_matrix, dbow_matrix))




"""
    Create an empty string of parameter length.
    Used in conjunction with python's translation method to, in this case, pre-process text
    for Doc2Vec.
"""
def create_whitespace(length):
    whitespace = ''

    for i in range(length):
        whitespace += ' '

    return whitespace



"""
    Simple method for finding and returning all file directoies in root and
    subroot directory.
"""
def get_all_files(path):
    file_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:

            # Make sure hidden files do not make into the list
            if name[0] == '.':
                continue
            file_paths.append(os.path.join(path, name))
    return file_paths

"""
    Returns all the folders in the specified path.
"""
def get_all_folders(path):
    folders = []

    for folder_path in os.listdir(path):
        folder_path = os.path.join(path, folder_path)
        if os.path.isdir(folder_path):
            folders.append(folder_path)

    return folders


def preprocess_line(line, tokenize=True):
    punctuation = "`~!@#$%^&*()_-=+[]{}\|;:'\"|<>,./?åαβ"
    numbers = "1234567890"
    number_replacement = create_whitespace(len(numbers))
    spacing = create_whitespace(len(punctuation))

    lowercase_line = line.lower()
    translation_table = str.maketrans(punctuation, spacing)
    translated_line = lowercase_line.translate(translation_table)
    translation_table_numbers = str.maketrans(numbers, number_replacement)
    final_line = translated_line.translate(translation_table_numbers)

    if tokenize:
        line_tokens = utils.to_unicode(final_line).split()

        return line_tokens
    else:
        return final_line

"""
    Extract all sentences from text provided, and then preprocess it. Each line is preprocessed by removing numbers
    and special characters.
"""
def preprocess_parse_sentences(text, tokenize):
    sentence_token_list = sent_tokenize(text)
    parsed_processed_sentences = []

    for sentence in sentence_token_list:
        preprocessed_sentence = preprocess_line(sentence, tokenize)
        parsed_processed_sentences.append(preprocessed_sentence)

    return parsed_processed_sentences


def preprocess(dir):
    pubmed_folders = os.listdir(dir)
    pubmed_folders.pop(0)
    punctuation = "`~!@#$%^&*()_-=+[]{}\|;:\"|<>,./?åαβ"
    numbers = "1234567890"
    number_replacement = create_whitespace(len(numbers))
    spacing = create_whitespace(len(punctuation))

    files = get_all_files(dir)

    output_file = open('preprocessed_text.txt', 'w')

    for i, file in enumerate(files):
        tokens = set([])
        text_file_object = open(file, 'r')


        for line in text_file_object:
            if 'Open Access' in line:
                break
            else:
                if line.strip() == '':
                    continue
                lowercase_line = line.lower()
                translation_table = str.maketrans(punctuation, spacing)
                translated_line = lowercase_line.translate(translation_table)
                translation_table_numbers = str.maketrans(numbers, number_replacement)
                final_line = translated_line.translate(translation_table_numbers)
                line_tokens = utils.to_unicode(final_line).split()
                tokens = list(set(tokens) | set(line_tokens))

                preprocessed_text = ''

        #Create string with all tokens to write to file
        for token in tokens:
            preprocessed_text = preprocessed_text + ' ' + token

        output_file.write(preprocessed_text + '\n')
        print('Document written: ', i)


def retrain(epochs, text_file, model_path, save_model_name):
    model = Doc2Vec.load(model_path)

    documents = LabeledLineDocument(text_file)

    for epoch in range(epochs):
        model.train(documents)
        shuffle(documents.n_list)

    model.save(save_model_name)


def start_training(text_file_name, model_file_name, epochs, vector_size, window_size, min_count, dm,
                   shuffle_lines=True):
    cores = multiprocessing.cpu_count()

    # Read preprocessed text file with gensim TaggedLineDocument
    lines = LabeledLineDocument(text_file_name, shuffle_lines=shuffle_lines)

    model = Doc2Vec(lines, size=vector_size, window=window_size, min_count=min_count, workers=cores, dm=dm)

    print('...training')

    for epoch in range(epochs):
        print('epoch: ', epoch + 1)

        model.train(lines)

        if shuffle_lines:
            shuffle(lines.n_list)

    model.save(model_file_name)


def write_documents_to_file(folder_path, output_file_name):
    files = get_all_files(folder_path)

    output_file = open(output_file_name, 'w')

    for file in files:

        text_file = open(file, 'r')
        output_file.write('---' + file + '---')

        for line in text_file:
            output_file.write(line)
        print("Wrote file: ", file)

