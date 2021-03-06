from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import zipfile
import tensorflow as tf
import numpy as np
# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = '/Users/Chip/data/'
FILE_NAME = 'text8.zip'
FILE_PATH = '/home/dingji/Desktop/new.txt'
# def download(file_name, expected_bytes):
#     """ Download the dataset text8 if it's not already downloaded """
#     file_path = DATA_FOLDER + file_name
#     if os.path.exists(file_path):
#         print("Dataset ready")
#         return file_path
#     file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
#     file_stat = os.stat(file_path)
#     if file_stat.st_size == expected_bytes:
#         print('Successfully downloaded the file', file_name)
#     else:
#         raise Exception('File ' + file_name +
#                         ' might be corrupted. You should try downloading it with a browser.')
#     return file_path

def read_data(file_path):
    """ Read data into a list of tokens
    There should be 17,005,207 tokens
    """
    with open(file_path) as f:
        words = tf.compat.as_str(f.read()).split()
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    for word, _ in count:
        dictionary[word] = index
        index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center word
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    file_path = FILE_PATH
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size),dictionary

def get_index_vocab(vocab_size):
    words = read_data(FILE_PATH)
    return build_vocab(words, vocab_size)
def main():
    dict = process_data(50000, 128,5)
if __name__ == '__main__':
    main()

