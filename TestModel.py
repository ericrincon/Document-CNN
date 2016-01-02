import os
import getopt
import numpy
from DocNet import DocNet

import ExperimentScript

def main():
    model_path = None
    test_folder_path = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:m:e:s:w:c:d:t:', ['model_path=', 'test_folder='])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-m', '--model_path'):
            text_file_path = arg
        elif opt in ('-t', '--test_folder'):

        else:
            sys.exit(2)
    assert model_path is not None, 'Enter a model path!'
    assert test_folder_path is not None, 'Enter a test folder path!'

    # Get test files
    print('...testing')
    X_test, Y_test = ExperimentScript.get_neg_pos(text_file_path, d2v_model_path, n_examples, doc_max)
    Y_test_array = numpy.zeros(Y_test.shape[0])
    Y_test_array[Y_test[:, 1] == 1] = 1

    accuracy, f1, auc, precision, recall = doc_cnn.test(X_test, Y_test_array)
    model = DocNet(model_path=model_path)
    model.test()


if __name__ == '__main__':
    main()