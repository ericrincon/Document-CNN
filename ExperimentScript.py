__author__ = 'ericrincon'

import Doc2VecTool
import getopt
import sys
import numpy

from gensim.models import Doc2Vec
from DocNet import DocNet
"""
    Script to test out DocNet on the IMDB sentiment analysis.
"""

def main():
    folder_path = None
    d2v_model_path = None
    n_examples = None
    n_epochs = 40
    learning_rate = .01
    mini_batch_size = 32
    momentum = .5
    learning_rate_decay = 1e-6
    test_folder = None
    doc_max = 50
    use_graph = True
    cnn_model_name = 'cnn.h5py'
    doc_vector_size = 100
    verbose = 1
    hidden_layers = [100]
    filters = [2, 3, 4, 5, 6]
    convolution_type = 2
    optimization_method = 'adagrad'
    headless_plot = False
    skipthought = False
    dropout = .5
    options = ['input_folder=', 'd2v_model_path=', 'n_examples=', 'n_epochs=', 'learning_rate=', 'mini_batch_size=',
               'momentum=', 'lr_decay=', 'help=', 'test_folder=', 'doc_max_size=', 'graph=', 'cnn_model_name=',
               'doc_vector_size=', 'verbose=', 'hidden_layers=', 'filter_sizes=', 'convolution_type=',
               'optimization_method=', 'headless_plot=', 'skipthoughts=', 'dropout=']
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:m:n:e:l:b:w:d:h:t:g:c:v:z:f:', options)
    except getopt.GetoptError:
        print('Error')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--input_folder'):
            folder_path = arg
        elif opt in ('-m', '--d2v_model_path'):
            d2v_model_path = arg
        elif opt in ('-n', '--n_examples'):
            n_examples = int(arg)
        elif opt in ('-e', '--n_epochs'):
            n_epochs = int(arg)
        elif opt in ('-l', '--learning_rate'):
            learning_rate = float(arg)
        elif opt in ('-b', '--mini_batch_size'):
            mini_batch_size = int(arg)
        elif opt in ('-w', '--momentum'):
            momentum = float(arg)
        elif opt in ('-d', '--lr_decay'):
            learning_rate_decay = float(arg)
        elif opt in ('-t', '--test_folder'):
            test_folder = arg
        elif opt in ('-c', '--cnn_model_name'):
            cnn_model_name = arg
        elif opt in ('-d', '--doc_vector_size'):
            doc_vector_size = int(arg)
        elif opt in ('-g', '--graph'):
            value = int(arg)

            if value == 0:
                use_graph = False

        elif opt in ('-h', '--help'):
            for command in options:
                print(command)
            sys.exit()
        elif opt in ('-s', '--doc_max_size'):
            doc_max = int(arg)
        elif opt in ('-v', '--verbose'):
            verbose = int(arg)
        elif opt in ('-z', '--hidden_layers'):
            hidden_layers = []
            sizes = arg.split(',')

            for size in sizes:
                hidden_layers.append(int(size))
        elif opt in ('-f', '--filter_sizes'):
            filters = []

            filters_sizes = arg.split(',')

            for filter in filters_sizes:
                filters.append(int(filter))
        elif opt == '--convolution_type':
            convolution_type = int(arg)
        elif opt == '--optimization_method':
            optimization_method = arg
        elif opt == '--headless_plot':
            option = int(arg)

            if option == 1:
                headless_plot = True
        elif opt == '--skipthoughts':
            option = int(arg)

            if option == 0:
                skipthought = False
        elif opt == '--dropout':
            dropout = float(arg)
        else:
            print('Error: {} not recognized'.format(opt))
            sys.exit(2)

    assert folder_path is not None, 'You must specify a folder!'
    assert d2v_model_path is not None, 'You must specify a d2v model path!'

    # Get training files
    print('...reading files')
    if convolution_type == 2 and not skipthought:
        X_train, Y_train = get_neg_pos(folder_path, d2v_model_path, n_examples, doc_max)
    elif convolution_type == 1 and not skipthought:
        X_train, Y_train = read_docs_file(folder_path, d2v_model_path, doc_vector_size, n_examples)
    elif skipthought:
        print('not...ready...')


    print('...creating model')
    doc_cnn = DocNet(doc_max_size=doc_max, n_feature_maps=2, graph=use_graph, doc_vector_size=doc_vector_size,
                     hidden_layer_sizes=hidden_layers, filter_sizes=filters, convolution=convolution_type,
                     dropout_p=dropout)

    print('...training')
    doc_cnn.train(X_train, Y_train, n_epochs=n_epochs, batch_size=mini_batch_size, learning_rate=learning_rate,
                  lr_decay=learning_rate_decay, momentum=momentum, nesterov=True, model_name=cnn_model_name,
                  verbose=verbose, optimization_method=optimization_method, plot_headless=headless_plot)

    # Get test files
    print('...testing')
    if convolution_type == 2:
        X_test, Y_test = get_neg_pos(test_folder, d2v_model_path, n_examples, doc_max)
    else:
        X_test, Y_test = read_docs_file(folder_path, d2v_model_path, doc_vector_size, n_examples)
    Y_test_array = numpy.zeros(Y_test.shape[0])
    Y_test_array[Y_test[:, 1] == 1] = 1

    accuracy, f1, auc, precision, recall = doc_cnn.test(X_test, Y_test_array)



def get_neg_pos(folder_path, d2v_model_path, examples_limit=None, sentence_limit=50):
    folders = Doc2VecTool.get_all_folders(folder_path)
    neg_folder = None
    pos_folder = None

    for folder in folders:
        if 'neg' in folder:
            neg_folder = folder
        elif 'pos' in folder:
            pos_folder = folder

    neg_examples = Doc2VecTool.get_all_files(neg_folder)
    pos_examples = Doc2VecTool.get_all_files(pos_folder)

    # Load Doc2vec model
    print('...loading Doc2Vec model')
    d2v_model = Doc2Vec.load(d2v_model_path)

    # Load the positive and negative examples
    print('...processing data')
    X_train_neg = read_files(neg_examples, d2v_model, examples_limit, sentence_limit)
    X_train_pos = read_files(pos_examples, d2v_model, examples_limit, sentence_limit)
    X_train = numpy.vstack((X_train_neg, X_train_pos))
    Y_train = numpy.zeros((X_train_neg.shape[0] + X_train_pos.shape[0], 2))
    Y_train[:X_train_neg.shape[0], 0] = 1
    Y_train[X_train_neg.shape[0] + 1:, 1] = 1

    return X_train, Y_train



def read_files(files, d2v_model, file_limit=None, sentence_limit=50):
    documents = []

    for i, file_path in enumerate(files):
        if file_limit:
            if i == (file_limit - 1):
                break

        lines = []
        file = open(file_path, 'r')

        for i, line in enumerate(file):
            if i == sentence_limit:
                break
            lines.append(d2v_model.infer_vector(line))

        document = numpy.zeros((sentence_limit, lines[0].shape[0]))

        for i, line in enumerate(lines):
            document[i, :] = line

        documents.append(document)

    # Keras convolution layer takes in a 4D tensor.
    # Format data into a 4D tensor where the dimensions
    # (number of examples, channels, max number of sentences, size of Doc2Vec vector)
    documents_tensor = numpy.zeros((len(documents), 1, sentence_limit, documents[0].shape[1]))

    for i, document in enumerate(documents):
        documents_tensor[i, :, :, :] = document

    return documents_tensor

def read_docs_file(file_path, d2v_model_path, d2v_vector_size, line_limit=None, steps=1):
    d2v_model = Doc2Vec.load(d2v_model_path)

    file = open(file_path, 'r')
    X_train = numpy.zeros((50000, steps, d2v_vector_size))
    Y_train = numpy.zeros((50000, 2))

    for i, line in enumerate(file):
        if i == 50000:
            break
        if line_limit:
            if i == line_limit:
                break

        X_train[i, :, :] = d2v_model.infer_vector(line)

        if i <= 24999:
            Y_train[i, 0] = 1
        else:
            Y_train[i, 1] = 1
    return X_train, Y_train


if __name__ == '__main__':
    main()