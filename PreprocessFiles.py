import sys
import getopt
import Doc2VecTool
import os
import nltk
import numpy
import h5py

from gensim.models import Word2Vec

"""
    Helper script that takes in all the IMDB reviews and writes them to one file. This file also write a new text file
    with the data preprocessed.
"""
w2v_model = None

def main():
    train_folders_path = ''
    train_output_file_name = ''
    train_folder_output_path = ''
    test_folders_path = ''
    test_folders_output_path = ''
    test_output_file_name = ''
    sentences = True
    w2v = True
    max_sentences = 10
    max_words = 20
    w2v_model_path = None
    w2v_size = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:o:p:s:w:m:', ['train_folders_path=', 'train_output_file_name=',
                                                              'train_folder_output_path=', 'test_folders_path=',
                                                              'test_output_file_name=', 'test_folders_output_path=',
                                                              'sentences=', 'w2v=', 'max_sentences=',
                                                              'w2v_model_path=', 'max_words=', 'w2v_size='])
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
        elif opt in ('-w', '--w2v'):
            option = int(arg)

            if option == 0:
                w2v = False
        elif opt in ('-m', '--max-sentences'):
            max_sentences = int(arg)
        elif opt == '--w2v_model_path':
            w2v_model_path = arg
        elif opt == '--max_words':
            max_words = arg
        elif opt == '--w2v_size':
            w2v_size = int(arg)
        else:
            print('No such arg: ', opt)
            sys.exit(2)
    print('...start')

    if w2v:
        assert w2v_model_path is not None, 'Provide a path to a w2v model!'
        assert w2v_size is not None, 'Provide a size for the w2v vector'

        preprocess_documents(train_folders_path, w2v_model_path, max_sentences, max_words, w2v_size, 'train.hdf5')
        preprocess_documents(test_folders_path, w2v_model_path, max_sentences, max_words, w2v_size, 'test.hdf5')

    else:

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
    text_file = open(file_path).read()
    parsed_processed_sentences = Doc2VecTool.preprocess_parse_sentences(text_file, False)

    sentence_to_write = parsed_processed_sentences.pop(0)

    for preprocesed_sent in parsed_processed_sentences:
        sentence_to_write = sentence_to_write + ' ' + preprocesed_sent
    all_sentences_file.write(sentence_to_write + '\n')


def preprocess_documents(folder, w2v_model_path, max_sentences, max_words, w2v_size, name=None):
    folders = Doc2VecTool.get_all_folders(folder)
    global w2v_model

    word_matrix = numpy.zeros((25000, 1, max_words, max_sentences * w2v_size))

    if not w2v_model:
        print('loading w2v model...')
        w2v_model = Word2Vec.load_word2vec_format(w2v_model_path, binary=True)


    neg_folder = None
    pos_folder = None

    for folder in folders:
        if 'neg' in folder:
            neg_folder = folder
        elif 'pos' in folder:
            pos_folder = folder

    folders = [neg_folder, pos_folder]

    for folder in folders:
        files = Doc2VecTool.get_all_files(folder)

        for doc_i, file in enumerate(files):
            file_body = open(file).read()
            sentences = nltk.sent_tokenize(file_body)
            sentences_matrix = numpy.zeros((max_words, w2v_size * max_sentences))

            for n_sentence, sentence in enumerate(sentences):
                if n_sentence == max_sentences:
                    break
                words = nltk.word_tokenize(sentence)

                #sentence_matrix = numpy.zeros((max_words, w2v_size))

                for i, word in enumerate(words):
                    if i == max_words:
                        break
                    try:
                        sentences_matrix[i, n_sentence * w2v_size: w2v_size  * (n_sentence + 1)] = w2v_model[word]
                    except KeyError:
                        continue

            sentences_matrix = numpy.reshape(sentences_matrix, (1, sentences_matrix.shape[0], sentences_matrix.shape[1]))
            word_matrix[doc_i, :, :, :] = sentences_matrix

    output = h5py.File(name)
    output.create_dataset('data', data=word_matrix)


if __name__ == '__main__':
    main()