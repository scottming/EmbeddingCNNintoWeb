# -*- coding: utf-8 -*-
import csv
import os
import pickle
import sqlite3

import click
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

from .text_cnn import TextCNN
from .text_helpers import (build_dataset, clean_data, jieba_cut, load_dict,
                           pad_crop, read_stopwords, word_to_number,
                           VOCAB2IX_PATH, IX2VOCAB_PATH, STOPWORDS_PATH)

app = Flask(__name__)


######## Preparing the Classifier
CUR_DIR = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CUR_DIR)
DB_PATH = os.path.join(CUR_DIR, 'reviews.sqlite')


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


def csv_entry(path, document, y, feedback):
    with open(path, 'a') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow([document, y, feedback])


def build_eval_dataset(data, vocab2ix_path, ix2vocab_path,
                       stopwords_path, max_words=20):
    # Read dict and stopwords
    vocab2ix, _ = load_dict(vocab2ix_path, ix2vocab_path)
    stopwords = read_stopwords(stopwords_path)
    # Clean eval data
    data_list = [data]
    data_cut = jieba_cut(data_list)
    text_data = clean_data(data_cut, stopwords)
    # Words to number
    num_data = word_to_number(text_data, vocab2ix)
    num_data_pad = list(pad_crop(num_data, max_words))
    return np.asarray(num_data_pad)


def get_pred_prob(eval_data):
    graph_path = os.path.join(PARENT_PATH, 'save_model/model.ckpt.meta')
    sess_path = os.path.join(PARENT_PATH, 'save_model/model.ckpt')
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph(graph_path)
            saver.restore(sess, sess_path)
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            keep_proba = graph.get_operation_by_name('keep_proba').outputs[0]
            y = graph.get_operation_by_name('output/y').outputs[0]
            pred = graph.get_operation_by_name('accuracy/pred').outputs[0]

            # Set feed_dict and get prediction
            feed_dict = {input_x: eval_data, keep_proba: 1.0}
            pred_eval = pred.eval(feed_dict=feed_dict)
            prob_eval = y.eval(feed_dict=feed_dict).max(axis=1)
    return pred_eval, prob_eval


def classify(pred_eval, prob_eval):
    label = {0: 'negative', 1: 'positive'}
    sentiment = label[pred_eval[0]]
    return sentiment, prob_eval[0]


######## Flask
class ReviewForm(Form):
    review_text = TextAreaField('',
                                [validators.DataRequired(),
                                 validators.length(min=15)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['review_text']
        evaldata = build_eval_dataset(
            review, VOCAB2IX_PATH, IX2VOCAB_PATH, STOPWORDS_PATH)
        pred_eval, prob_eval = get_pred_prob(evaldata)
        y, proba = classify(pred_eval, prob_eval)
        return render_template('results.html',
                               content=review,
                               prediction=y,
                               probability=round(proba * 100, 2))
    return render_template('reviewform.html', form=form)


@app.route('/thanks', methods=['POST'])
def feedback():
    csv_path = os.path.join(CUR_DIR, 'reviews.csv')
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    sqlite_entry(DB_PATH, review, y)
    csv_entry(csv_path, review, y, feedback)
    return render_template('thanks.html')


@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--debug', is_flag=True)
def cli(host, debug):
    app.run(host=host, debug=debug)


if __name__ == '__main__':
    cli()
