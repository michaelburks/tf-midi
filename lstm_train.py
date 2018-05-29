import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_io

import json
import os

import console
from constants import *

import metrics
import midi_io
import midi_coder


def midi_files(dir):
    files = []
    for root, _, filenames in os.walk(dir):
        for f in filenames:
            if f.endswith('.mid'):
                files.append(root + f)
    return files


def featurize(data):
    data_len, encoded_size = data.shape
    inputs = []
    outputs = []

    for i in range(0, data_len - WINDOW - 1, STEP):
        inputs.append(data[i:i+WINDOW])
        outputs.append(data[i+WINDOW])

    sample_count = len(inputs)

    train_data = np.array(inputs, np.float32).reshape((sample_count, WINDOW, encoded_size))
    target_data = np.array(outputs, np.float32).reshape((sample_count, encoded_size))

    return train_data, target_data


def train_model(datafiles):
    sample_len = WINDOW
    feat_len = midi_coder.MIDI_COUNT

    with tf.name_scope('input'):
        # placeholders
        x = tf.placeholder(tf.float32, shape=(None, sample_len, feat_len), name='x-input')
        y = tf.placeholder(tf.float32, shape=(None, feat_len), name='y-input')

        # it = tf.placeholder(tf.bool, shape = (1,), name='is_training')
        # Add a histogram of the different labels that are passed in for each iteration. This helps
        # determine how evenly distributed labels are across epochs.
        tf.summary.histogram('notes', y[:, 0])

    weight = tf.Variable(tf.random_normal([LSTM_SIZE, feat_len]))
    bias = tf.Variable(tf.random_normal([feat_len]))

    lstm_cell = tf.contrib.rnn.LSTMCell(LSTM_SIZE, forget_bias=1.0)
    cell = lstm_cell

    # add a dropout wrapper if training
    # if is_training and dropout < 1:
    #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

    input = tf.transpose(x, [1, 0, 2])
    input = tf.reshape(input, [-1, feat_len])
    input = tf.split(input, WINDOW, 0)

    outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
    y_prediction = tf.matmul(outputs[-1], weight) + bias

    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(logits=y_prediction, name='prediction')
        tf.summary.histogram('predicted', prediction)

    # Compute the loss of the iteration.
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_prediction)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', cross_entropy)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARN_RATE).minimize(cross_entropy)

    with tf.name_scope('evaluation'):
        eval, stats = metrics.class_metrics(prediction, y, midi_coder.classes())

    summ = tf.summary.merge_all()

    sess = tf.InteractiveSession()

    train_writer = tf.summary.FileWriter(OUTPUT_DIR, sess.graph)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    epoch_metrics = []
    metric_summaries = []

    for i in range(EPOCH_COUNT):
        print "----------- Epoch {0}/{1} -----------".format(i+1, EPOCH_COUNT)
        percentage = 0
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for df in datafiles:
            print df
            data =  midi_coder.encode_list(midi_io.read_midi(df))
            train_data, target_data = featurize(data)

            batched = False
            if (batched):
                num_batches = int(len(train_data)/BATCH_SIZE)

                for j in range(num_batches):
                    batch_start = j * BATCH_SIZE
                    train_batch = train_data[batch_start:batch_start+BATCH_SIZE]
                    target_batch = target_data[batch_start:batch_start+BATCH_SIZE]
                    run_metadata = tf.RunMetadata()
                    summary = sess.run([optimizer],
                                       feed_dict={x:train_batch, y:target_batch},
                                       options=run_options,
                                       run_metadata=run_metadata)
                    console.progress(j, num_batches-1, "batch {0}/{1}".format(j+1, num_batches))
                print ""
            else:
                run_metadata = tf.RunMetadata()
                summary = sess.run([optimizer],
                                   feed_dict={x:train_data, y:target_data},
                                   options=run_options,
                                   run_metadata=run_metadata)

        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % i)

        eval_x = train_data[0:100]
        eval_y = target_data[0:100]
        eval_dict = {x:eval_x, y:eval_y}

        o, p, e, st, sm = sess.run([outputs, prediction, eval, stats, summ], feed_dict=eval_dict)

        train_writer.add_summary(sm, i)

        clean_st = metrics.sanitize(st)
        epoch_metrics.append(metrics.sanitize(clean_st))
        summ_st = metrics.summarize(clean_st)
        metric_summaries.append(summ_st)
        print summ_st

    graph_out = OUTPUT_DIR + 'graph'
    print "Exporting trained model to:", graph_out

    graph_io.write_graph(sess.graph, graph_out, 'graph')
    saver = tf.train.Saver(tf.trainable_variables())
    saver.save(sess, graph_out, write_meta_graph=True)
    print "Successfully saved model"

    results = {
        'config': ConfigSummary(),
        'epoch_metrics': epoch_metrics
    }
    with open(OUTPUT_DIR + 'result.json', 'w') as fp:
        fp.write(json.dumps(results))
        fp.close()

    with open(OUTPUT_DIR + 'summaries.json', 'w') as fp:
        fp.write(json.dumps(metric_summaries))
        fp.close()

    sess.close()


if __name__ == "__main__":
    print "Reading files from %s..." % INPUT_DIR
    datafiles = midi_files(INPUT_DIR)

    if tf.gfile.Exists(OUTPUT_DIR):
        tf.gfile.DeleteRecursively(OUTPUT_DIR)

    print "Training model..."
    train_model(datafiles)
