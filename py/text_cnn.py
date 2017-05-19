# -*- coding: utf-8 -*-
import tensorflow as tf


class TextCNN(object):
    """A CNN for text classification.
    """
    def __init__(self,
                 sequence_length,
                 vocab_size,
                 word_embed_size,
                 filter_sizes,
                 filter_num,
                 num_classes):

        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(
            tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.int32, shape=[None, ], name='input_y')
        self.keep_proba = tf.placeholder(
            tf.float32, shape=None, name='keep_proba')

        # Embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.get_variable('word_embedding',
                                     [vocab_size, word_embed_size],
                                     tf.float32,
                                     tf.random_normal_initializer())
            self.embeds = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embeds_expanded = tf.expand_dims(self.embeds, -1)

        # Convolution + maxpool layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope(f'conv-maxpool-{filter_size}'):
                filter_shape = [filter_size, word_embed_size, 1, filter_num]
                W = tf.get_variable(f"W-{filter_size}",
                                    filter_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable(f"b-{filter_size}", [filter_num],
                                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(self.embeds_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name=f'conv-{filter_size}')
                conv_hidden = tf.nn.tanh(tf.add(conv, b), name=f'tanh-{filter_size}')
                # conv_hidden = tf.nn.relu(tf.add(conv, b), name=f'relu-{filter_size}')
                pool = tf.nn.max_pool(conv_hidden,
                                      ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='VALID',
                                      name=f'pool-{filter_size}')
                pooled_outputs.append(pool)

            num_filters_total = filter_num * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Drop out layer
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_proba)

        # Final scores and predictions
        with tf.name_scope('output'):
            softmax_w = tf.get_variable('softmax_w', [num_filters_total, num_classes],
                                        tf.float32, tf.random_normal_initializer())
            softmax_b = tf.get_variable('softmax_b', [num_classes], tf.float32,
                                        tf.constant_initializer(0.0))
            self.logits = tf.matmul(self.h_drop, softmax_w) + softmax_b
            self.y = tf.nn.softmax(self.logits, name='y')

        # CalculateMean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y, name='losses')
            self.loss = tf.reduce_mean(losses, name='loss')

        # Accuracy
        with tf.name_scope('accuracy'):
            self.pred = tf.argmax(self.y, 1, name='pred')
            correct_prediction = tf.equal(tf.cast(self.pred, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                           name='accuracy')
