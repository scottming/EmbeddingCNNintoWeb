# coding: utf-8
from collections import Counter
import itertools
import os
import operator
import pickle
import sqlite3
import string
import time

import jieba
import more_itertools
import numpy as np
import zhon.hanzi as zh


CUR_DIR = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CUR_DIR)
VOCAB2IX_PATH = os.path.join(PARENT_PATH, 'data', 'vocab2ix.pkl')
IX2VOCAB_PATH = os.path.join(PARENT_PATH, 'data', 'ix2vocab.pkl')
STOPWORDS_PATH = os.path.join(PARENT_PATH, 'data', 'stop_words_chinese.txt')


def read_stopwords(filename):
    with open(filename, 'r') as file:
        stopwords = file.read().splitlines()
    return stopwords


def read_data(filename):
    with open(filename) as file:
        sequences = file.read().splitlines()
    for sequence in sequences:
        line = sequence.split('\t')
        yield line


def jieba_cut(sequences):
    jieba.setLogLevel(20)
    jieba.enable_parallel(8)
    for sequence in sequences:
        data = jieba.cut(sequence)
        yield ' '.join(data)


def clean_data(sequences, stopwords):
    sequences = (''.join(c for c in x if c not in string.punctuation)
                           for x in sequences)
    sequences = (''.join(c for c in x if c not in zh.punctuation)
                           for x in sequences)
    sequences = (x.split() for x in sequences)
    sequences = ([c for c in x if c] for x in sequences)
    sequences = ([c for c in x if c not in stopwords] for x in sequences)
    return sequences


def get_common_words(words, n):
    count = Counter(words)
    count_dict = {key: value for key, value in count.items() if value > n}
    word_counts = sorted(count_dict.items(),
                         key=operator.itemgetter(1), reverse=True)
    return word_counts


def build_dict(word_counts):
    count = [['<UNK>', -1]]
    count.extend(word_counts)
    vocab2ix = {key: ix for ix, (key, _) in enumerate(count)}
    ix2vocab = {value: key for key, value in vocab2ix.items()}
    return vocab2ix, ix2vocab


def word_to_number(sequences, word_dict):
    data = []
    for sequence in sequences:
        sequence_data = []
        for word in sequence:
            try:
                sequence_data.append(word_dict[word])
            except:
                sequence_data.append(0)
        data.append(sequence_data)
    return data


def pad_crop(sequences, n):
    for sequence in sequences:
        sequence = more_itertools.padded(
            sequence, fillvalue=0, n=n)
        sequence = list(itertools.islice(sequence, n))
        yield sequence


def save_dict(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_dict(vocab2ix_path, ix2vacab_path):
    with open(vocab2ix_path, 'rb') as f1, open(ix2vacab_path, 'rb') as f2:
        vocab2ix = pickle.load(f1)
        ix2vocab = pickle.load(f2)
    return vocab2ix, ix2vocab


def get_clean_data(path, stopwords):
    data = list(zip(*read_data(path)))
    sentiment = data[1]
    cut_words = jieba_cut(data[0])
    # 流向两个地方，1 是制造 word_dict, 2 是转为 numbers
    clean_words = list(clean_data(cut_words, stopwords))
    return clean_words, sentiment


def build_dataset(train_path, test_path, stopwords_path, n=10, max_words=20):
    """A function to build dataset and save them."""
    print('Building dataset...')
    # Get words and sentiment
    start = time.time()
    stopwords = read_stopwords(stopwords_path)
    words_train, sentiment_train = get_clean_data(train_path, stopwords)
    words_test, sentiment_test = get_clean_data(test_path, stopwords)
    data = itertools.chain(words_train, words_test)
    words = itertools.chain.from_iterable(data)

    # Words to numbers
    word_counts = get_common_words(words, n)
    vocab2ix, ix2vocab = build_dict(word_counts)
    save_dict(vocab2ix, VOCAB2IX_PATH)
    save_dict(ix2vocab, IX2VOCAB_PATH)
    text_train = word_to_number(words_train, vocab2ix)
    text_test = word_to_number(words_test, vocab2ix)

    # Pad/crop sequence
    text_data_train = np.array(
        list(pad_crop(text_train, max_words)))
    text_data_test = np.array(
        list(pad_crop(text_test, max_words)))

    # Get sentiment data
    sentiment_data_train = np.fromiter(sentiment_train, int).reshape((-1, 1))
    sentiment_data_test = np.fromiter(sentiment_test, int).reshape((-1, 1))

    # Concat text and sentiment
    train_data = np.concatenate((text_data_train, sentiment_data_train), axis=1)
    test_data = np.concatenate((text_data_test, sentiment_data_test), axis=1)

    # Save data
    train_data_path = os.path.join(PARENT_PATH, 'data', 'train_data.txt')
    test_data_path = os.path.join(PARENT_PATH, 'data', 'test_data.txt')
    np.savetxt(train_data_path, train_data, fmt='%d')
    np.savetxt(test_data_path, test_data, fmt='%d')
    stop = time.time()
    print(f'Wasted {stop-start:.2f} seconds, Done!!!')


def sqltext_to_number(
        X, vocab2ix_path=VOCAB2IX_PATH,
        ix2vocab_path=IX2VOCAB_PATH,
        stopwords_path=STOPWORDS_PATH,
        max_words=20):
    # Read dict and stopwords
    vocab2ix, _ = load_dict(vocab2ix_path, ix2vocab_path)
    stopwords = read_stopwords(stopwords_path)
    # Clean data
    data_cut = jieba_cut(X)
    text_data = clean_data(data_cut, stopwords)
    # Words to numbers
    num_data = word_to_number(text_data, vocab2ix)
    num_data_pad = list(pad_crop(num_data, max_words))
    return np.asarray(num_data_pad)


def get_continue_train_dataset(db_path, continue_train_size):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM review_db ORDER BY date DESC')
    results = c.fetchmany(continue_train_size)
    data = np.array(results)
    X = data[:, 0]
    y = data[:, 1].astype(int)
    x_train = sqltext_to_number(X)
    conn.close()
    return x_train, y
