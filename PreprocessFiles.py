import sys
import getopt
import Doc2VecTool
import os

"""
    Helper script that takes in all the IMDB reviews and writes them to one file. This file also write a new text file
    with the data preprocessed.
"""


def main():
    folders_sub_path = ''
    output_file_name = ''
    folder_output_path = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:o:p:', ['folders_path', 'output_file_name', 'folder_output_path'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-f', '--folders_path'):
            folders_sub_path = arg
        elif opt in ('-o', 'output_file_name'):
            output_file_name = arg
        elif opt in ('-p', '--folder_output_path'):
            folder_output_path = arg
        else:
            sys.exit(2)
    print('...start')

    paths = ['neg/', 'pos/', 'unsup/']
    updates_paths = []

    for path in paths:
        updates_paths.append(folders_sub_path + path)

    all_sentences_file = open(folder_output_path + output_file_name, 'w+')

    for path in updates_paths:
        files = Doc2VecTool.get_all_files(path)
        for file_path in files:
            print('writing file ' + file_path)
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
                all_sentences_file.write(sentence + '\n')
                preprocessed_text_file.write(sentence + '\n')

            # Write an end of document tag once all sentences from a file have been written.
            # This is so that the doc2vec model does not predict a sentence that does not correspond to it's respective
            # file.
            all_sentences_file.write('<eod>.\n')


if __name__ == '__main__':
    main()