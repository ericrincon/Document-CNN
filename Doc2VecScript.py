__author__ = 'ericrincon'

import Doc2VecTool
import sys
import getopt

def main():
    # Get args for running

    text_file_path = ''
    model_name = ''
    epochs = 20
    d2v_vector_size = 100
    window_size = 10
    min_count = 0
    dm = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:m:s:w:m:d:', ['input_file=', 'model_name=', 'epochs=',
                                                                  'd2v_vector_size=', 'window_size=', 'min_count=',
                                                                  'dm'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-f', '--input_file'):
            text_file_path = arg
        elif opt in ('-m', '--modle_name'):
            model_name = arg
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('s', '--d2v_vector_size'):
            d2v_vector_size = int(arg)
        elif opt in ('w', '--window_size'):
            window_size = int(arg)
        elif opt in ('m', '--min_count'):
            min_count = int(arg)
        elif opt in ('d', '--dm'):
            dm = int(arg)
        else:
            sys.exit(2)

    Doc2VecTool.start_training(text_file_path, model_name, epochs, d2v_vector_size, window_size, min_count, dm)




if __name__ == '__main__':
    main()