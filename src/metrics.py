#!/usr/local/bin/python
import tensorflow as tf
import numpy as np
import math


def n_accurate(y, y_):
    """
    Computes the number of accurate predictions.
    y, y_: Tensor, shape: [batch_size, y_size].
    """
    correct_y_batch = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    n_accurate = tf.reduce_sum(tf.cast(correct_y_batch, tf.float32))  # Similar to numpy.count_nonzero().
    return n_accurate


def eval_acc(n_acc, total):
    """
    Evaluates the accuracy.
    """
    return float(n_acc) / total if total > 0 else 0.0


def create_confusion_matrix(y, y_, is_distribution=True):
    """
    Creates a confusion matrix for evaluation.
    By batch. Shape: [batch_size, y_size].
    """
    y, y_ = tf.convert_to_tensor(y), tf.convert_to_tensor(y_)
    n_samples = tf.cast(tf.shape(y_)[0], tf.float32)  # Get number of samples.
    
    if is_distribution:
        label_ref = tf.argmax(y_, axis=1)  # Ground truth labels.
        label_hyp = tf.argmax(y, axis=1)  # Predicted labels.
    else:
        label_ref, label_hyp = y, y_

    # Positive and negative in prediction.
    p_in_hyp = tf.reduce_sum(tf.cast(label_hyp, tf.float32))
    n_in_hyp = n_samples - p_in_hyp

    # True positives and false positives.
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_ref, 1), tf.equal(label_hyp, 1)), tf.float32))
    fp = p_in_hyp - tp

    # True negatives and false negatives.
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_ref, 0), tf.equal(label_hyp, 0)), tf.float32))
    fn = n_in_hyp - tn

    return tp.numpy(), fp.numpy(), tn.numpy(), fn.numpy()


def eval_mcc(tp, fp, tn, fn):
    """
    Evaluates the Matthews Correlation Coefficient (MCC).
    """
    core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else None


def eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=None):
    """
    Evaluates the results including accuracy, loss, and optionally MCC.
    """
    gen_acc = eval_acc(n_acc=gen_n_acc, total=gen_size)
    gen_loss = np.average(gen_loss_list)
    results = {'loss': gen_loss, 'acc': gen_acc}

    if use_mcc:
        gen_y = np.vstack(y_list)
        gen_y_ = np.vstack(y_list_)
        tp, fp, tn, fn = create_confusion_matrix(y=gen_y, y_=gen_y_)
        results['mcc'] = eval_mcc(tp, fp, tn, fn)

    return results


def basic_train_stat(train_batch_loss_list, train_epoch_n_acc, train_epoch_size):
    """
    Computes basic training statistics like average loss and accuracy.
    """
    train_epoch_loss = np.average(train_batch_loss_list)
    train_epoch_acc = eval_acc(n_acc=train_epoch_n_acc, total=train_epoch_size)
    return train_epoch_loss, train_epoch_acc
