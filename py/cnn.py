# -*- coding: utf-8 -*-
import os
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm

from .text_cnn import TextCNN
from .text_helpers import build_dataset, get_continue_train_dataset


CUR_DIR = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CUR_DIR)
DB_PATH = os.path.join(CUR_DIR, 'reviews.sqlite')
SESS_PATH = os.path.join(PARENT_PATH, 'save_model', 'model.ckpt')
GRAPH_PATH = os.path.join(PARENT_PATH, 'save_model', 'model.ckpt.meta')
VOCAB2IX_PATH = os.path.join(PARENT_PATH, 'data', 'vocab2ix.pkl')
IX2VOCAB_PATH = os.path.join(PARENT_PATH, 'data', 'ix2vocab.pkl')
TEST_DATA_PATH = os.path.join(PARENT_PATH, 'data', 'test_data.txt')
STOPWORDS_PATH = os.path.join(PARENT_PATH, 'data', 'stop_words_chinese.txt')
TRAIN_DATA_PATH = os.path.join(PARENT_PATH, 'data', 'train_data.txt')


def get_dataset(train_data_path, test_data_path):
    train = np.loadtxt(train_data_path, dtype=int)
    test = np.loadtxt(test_data_path, dtype=int)
    train_shuffle_idx = np.random.permutation(train.shape[0])
    test_shuffle_idx = np.random.permutation(test.shape[0])
    train = train[train_shuffle_idx]
    test = test[test_shuffle_idx]
    x_train = train[:, :-1]
    y_train = train[:, -1:].reshape((-1,))
    x_test = test[:, :-1]
    y_test = test[:, -1:].reshape((-1,))
    return (x_train, y_train), (x_test, y_test)


class Parameters():
    """Parameters for command(clean, train, eval...)."""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path


@click.group()
@click.option('--train-path', default=TRAIN_DATA_PATH,
              help='Default: data/train_data.txt.')
@click.option('--test-path', default=TEST_DATA_PATH,
              help='Default: data/test_data.txt.')
@click.pass_context
def cli(ctx, train_path, test_path):
    """CNN for Text Classification in Tensorflow.

    Examples:

    \b
        cnn train

    \b
        cnn train --confusion-matrix  # plot confusion matrix

    \b
        cnn --train-path train_shuffle.txt --test-path test_shuffle.txt clean  # text clean
    """
    ctx.obj = Parameters(train_path, test_path)


@cli.command()
@click.option('--stopwords-path', default=STOPWORDS_PATH)
@click.option('--sequence-length', default=20)
@click.option('-n', default=10,
              help='Find the common words that count is over n.')
@click.pass_obj
def clean(ctx, n, stopwords_path, sequence_length):
    build_dataset(ctx.train_path, ctx.test_path,
                  stopwords_path, n, sequence_length)


@cli.command()
@click.option('--vocab-size', default=80000)
@click.option('--num-classes', default=2)
@click.option('--filter-num', default=64)
@click.option('--batch-size', default=50)
@click.option('--word-embed-size', default=128)
@click.option('--training-steps', default=101)
@click.option('--learning-rate', default=0.01)
@click.option('--print-loss-every', default=20)
@click.option('--confusion-matrix', is_flag=True)
@click.option('--keep-proba', default=0.5)
@click.option('--filter-sizes', default=[3, 4, 5])
@click.option('--save-model', is_flag=True)
@click.pass_obj
def train(ctx, vocab_size, num_classes, filter_num,
          batch_size, word_embed_size, training_steps,
          learning_rate, print_loss_every, confusion_matrix,
          keep_proba, filter_sizes, save_model):

    # Load dataset
    (x_train, y_train), (x_test, y_test) = get_dataset(
        ctx.train_path, ctx.test_path)
    sequence_length = x_train.shape[1]
    dataset_size = x_train.shape[0]

    tf.reset_default_graph()
    with tf.Graph().as_default():
        cnn = TextCNN(sequence_length,
                      vocab_size,
                      word_embed_size,
                      filter_sizes,
                      filter_num,
                      num_classes)

        # Set eval feed_dict
        train_feed_dict = {cnn.input_x: x_train,
                           cnn.input_y: y_train,
                           cnn.keep_proba: 1.0}
        test_feed_dict = {cnn.input_x: x_test,
                          cnn.input_y: y_test,
                          cnn.keep_proba: 1.0}

        # Train
        saver = tf.train.Saver()
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(training_steps):
                start = (i * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                feed_dict={cnn.input_x: x_train[start:end],
                           cnn.input_y: y_train[start:end],
                           cnn.keep_proba: keep_proba}
                sess.run(train_step, feed_dict=feed_dict)
                if i % print_loss_every == 0:
                    avg_cost = cnn.loss.eval(feed_dict=feed_dict)
                    train_acc = cnn.accuracy.eval(feed_dict=train_feed_dict)
                    test_acc = cnn.accuracy.eval(feed_dict=test_feed_dict)
                    test_pred = cnn.pred.eval(feed_dict=test_feed_dict)
                    print(f"Epoch: {i:04d} | AvgCost: {avg_cost:7.4f}", end="")
                    print(f" | Train/Test ACC: {train_acc:.3f}/{test_acc:.3f}")

            # After training, save the sess
            if save_model:
                save_path = saver.save(sess, SESS_PATH)
                print('Model state has been saved!')

        if confusion_matrix:
            binary = cm(
                y_true=y_test, y_pred=test_pred
            )
            print('\n', 'Confusion Matrix: ')
            print(binary)
            plot_confusion_matrix(binary)
            plt.show()


@cli.command()
@click.option('--db-path', default=DB_PATH)
@click.option('--continue-train-size', default=10000)
@click.option('--batch-size', default=50)
@click.option('--training-steps', default=101)
@click.option('--learning-rate', default=0.01)
@click.option('--print-loss-every', default=5)
@click.option('--keep-proba', default=0.5)
@click.option('--save-model', is_flag=True)
@click.pass_obj
def continue_train(ctx, db_path, batch_size, training_steps,
                   learning_rate, print_loss_every, keep_proba,
                   continue_train_size, save_model):

    # Load whole dateset
    (x_train, y_train), (x_test, y_test) = get_dataset(
        ctx.train_path, ctx.test_path)

    # Load data for continuing train
    x_con_train, y_con_train = get_continue_train_dataset(db_path, 10000)
    dataset_size = x_con_train.shape[0]

    # Continue train
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph(GRAPH_PATH)
            saver.restore(sess, SESS_PATH)
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            input_y = graph.get_operation_by_name('input_y').outputs[0]
            keep_proba_ph = graph.get_operation_by_name('keep_proba').outputs[0]
            accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
            loss = graph.get_operation_by_name('loss/loss').outputs[0]

            # Initialize opt variables
            temp = set(tf.global_variables())
            train_step = tf.train.AdamOptimizer(learning_rate, name='adam2').minimize(loss)
            sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

            # Set eval feed_dict and print previous accuracy
            train_feed_dict = {input_x: x_train, input_y: y_train, keep_proba_ph: 1.0}
            test_feed_dict = {input_x: x_test, input_y: y_test, keep_proba_ph: 1.0}
            previous_train_acc = accuracy.eval(feed_dict=train_feed_dict)
            previous_test_acc = accuracy.eval(feed_dict=test_feed_dict)
            print("Previous: Train/Test ACC: ", end="")
            print(f"{previous_train_acc:.3f}/{previous_test_acc:.3f}")

            # Train
            for i in range(training_steps):
                start = (i * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                feed_dict={input_x: x_con_train[start:end],
                           input_y: y_con_train[start:end],
                           keep_proba_ph: keep_proba}
                sess.run(train_step, feed_dict=feed_dict)
                if i % print_loss_every == 0:
                    avg_cost = loss.eval(feed_dict=feed_dict)
                    train_acc = accuracy.eval(feed_dict=train_feed_dict)
                    test_acc = accuracy.eval(feed_dict=test_feed_dict)
                    print(f"Epoch: {i:04d} | AvgCost: {avg_cost:7.4f}", end="")
                    print(f" | Train/Test ACC: {train_acc:.3f}/{test_acc:.3f}")

            # After training, save the sess
            if save_model:
                save_path = saver.save(sess, SESS_PATH)
                print('Model state has been saved!')


if __name__ == '__main__':
    cli()
