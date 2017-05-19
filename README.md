![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.1.0-blue.svg)

# Embedding a Deep Learning Model into a Web Application


## Install

```
$ pip install -e .
```

## Usage


### Help

```
$ cnn --help
Usage: cnn [OPTIONS] COMMAND [ARGS]...

  CNN for Text Classification in Tensorflow.

  Examples:

      cnn train

      cnn train --confusion-matrix  # plot confusion matrix

      cnn --train-path train_shuffle.txt --test-path test_shuffle.txt clean  # text clean

Options:
  --train-path TEXT  Default: data/train_data.txt.
  --test-path TEXT   Default: data/test_data.txt.
  --help             Show this message and exit.

Commands:
  clean
  continue_train
  train
```

### Build dataset for training

```
$ cnn --train-path ../data/train_shuffle.txt --test-path ../data/test_shuffle.txt clean
```

### Train

```
$ cnn train
```

### APP

```
$ cnn_app --debug
```

### Continue train

This command is for Online-learning.

```
$ cnn continue_train
```









