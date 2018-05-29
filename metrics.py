import tensorflow as tf
import math

from midi_coder import decode_tensor

def class_metrics(predictions, labels, classes):
    """
        Creates per-class metrics. Ex) accuracy, precision, recall, and f1 score.
    """
    pred_idx = tf.argmax(predictions)
    label_idx = tf.argmax(labels)
    ops = []

    stats = {}

    for idx, label in enumerate(classes):
        with tf.name_scope(label):
            # For each class map the predictions and labels to whether each value matches the current
            # label. ex) label: 2, [2, 3, 5, 2, 3] -> [True, False, False, True, False]
            true_label = tf.map_fn(lambda l: tf.equal(l, idx), label_idx, dtype=tf.bool)
            true_pred = tf.map_fn(lambda l: tf.equal(l, idx), pred_idx, dtype=tf.bool)

            # Calculate precision and add a tensorboard scalar.
            precision, prec_op = tf.metrics.precision(labels=true_label, predictions=true_pred, name='p')
            tf.summary.scalar('precision', precision)

            # Calculate recall and add a tensorboard scalar.
            recall, rec_op = tf.metrics.recall(labels=true_label, predictions=true_pred, name='r')
            tf.summary.scalar('recall', recall)

            # Compute the f1 score and add a tensorboard scalar.
            f1 = (2 * recall * precision) / (recall + precision)
            tf.summary.scalar('f1', f1)

            # Calculate class accuracy and add a tensorboard scalar.
            accuracy = tf.reduce_mean(tf.cast(tf.equal(true_label, true_pred), tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            # Append the class operations to the ops list.
            ops.append(tf.group(rec_op, prec_op, accuracy, f1))
            stats[label] = {'p':precision, 'r':recall, 'f1':f1, 'a':accuracy}

    return tf.group(*ops), stats


def sanitize(stats):
    clean_stats = {}
    for note, note_stats in stats.items():
        clean_dict = {}
        for k, v in note_stats.items():
            c = float(v)
            if math.isnan(c):
                c = 0.0
            clean_dict[k] = c
        clean_stats[note] = clean_dict
    return clean_stats


def summarize(stats):
    final = {'p':0.0, 'r':0.0, 'f1':0.0, 'a':0.0}
    count = 0.0
    for note, note_stats in stats.items():
        count += 1.0
        for k, v in note_stats.items():
             final[k] += v
    for k in final:
        final[k] /= count
    return final
