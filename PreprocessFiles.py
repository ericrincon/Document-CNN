import sys
import getopt
import Doc2VecTool
import os

"""
    Helper script that takes in all the IMDB reviews and writes them to one file. This file also write a new text file
    with the data preprocessed.
"""


def main():
    train_folders_path = ''
    train_output_file_name = ''
    train_folder_output_path = ''
    test_folders_path = ''
    test_folders_output_path = ''
    test_output_file_name = ''
    sentences = True

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:o:p:s:', ['train_folders_path=', 'train_output_file_name=',
                                                              'train_folder_output_path=', 'test_folders_path=',
                                                              'test_output_file_name=', 'test_folders_output_path=',
                                                              'sentences='])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-f', '--train_folders_path'):
            train_folders_path = arg
        elif opt in ('-o', '--train_output_file_name'):
            train_output_file_name = arg
        elif opt in ('-p', '--train_folder_output_path'):
            train_folder_output_path = arg
        elif opt == '--test_folders_path':
            test_folders_path = arg
        elif opt == '--test_output_file_name':
            test_output_file_name = arg
        elif opt == '--test_folders_output_path':
            test_folders_output_path = arg
        elif opt in ('-s', '--sentences'):
            option = int(arg)

            if option == 0:
                sentences = False
        else:
            print('No such arg: ', opt)
            sys.exit(2)
    print('...start')

    paths = ['neg/', 'pos/', 'unsup/']
    updates_paths = []

    for path in paths:
        updates_paths.append(train_folders_path + path)

    all_sentences_file = open(train_folder_output_path + train_output_file_name, 'w+')

    for path in updates_paths:
        files = Doc2VecTool.get_all_files(path)

        for file_path in files:
            if sentences:
                preprocess_sentences(file_path, path, train_folder_output_path, all_sentences_file)
            else:
                preprocess_documents(file_path, all_sentences_file)
    # Handle test files

    paths = ['neg/', 'pos/']

    if not sentences:
        test_sentence_file = open(test_output_file_name, 'w+')

    for path in paths:
        updated_path = test_folders_path + path

        files = Doc2VecTool.get_all_files(updated_path)

        for file_path in files:
            if sentences:
                preprocess_sentences(file_path, updated_path, test_folders_output_path)
            else:
                preprocess_documents(file_path, test_sentence_file)



def preprocess_sentences(file_path, path, folder_output_path, all_sentences_file=None):
    print('writing file from ' + file_path)
    length = len(path.split('/'))
    new_folder = folder_output_path + path.split('/')[length-2] + '/'

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        print('create folder ' + new_folder)
    text_file = open(file_path).read()
    parsed_processed_sentences = Doc2VecTool.preprocess_parse_sentences(text_file, False)

    # Write the preprocessed sentences to one giant file and a new file
    preprocessed_text_file = open(new_folder + file_path.split('/')[-1], 'w+')

    for sentence in parsed_processed_sentences:
        if all_sentences_file is not None:
            all_sentences_file.write(sentence + '\n')
        preprocessed_text_file.write(sentence + '\n')

    # Write an end of document tag once all sentences from a file have been written.
    # This is so that the doc2vec model does not predict a sentence that does not correspond to it's respective
    # file.
    if all_sentences_file is not None:
        all_sentences_file.write('<eod>.\n')


def preprocess_documents(file_path, all_sentences_file):
    print('writing_document to master file')
    text_file = open(file_path, 'r')
    parsed_processed_sentences = Doc2VecTool.preprocess_parse_sentences(text_file, False)

    sentence_to_write = parsed_processed_sentences.pop(0)

    for preprocesed_sent in parsed_processed_sentences:
        sentence_to_write = sentence_to_write + ' ' + preprocesed_sent
    all_sentences_file.write(sentence_to_write + '\n')


if __name__ == '__main__':
    main()