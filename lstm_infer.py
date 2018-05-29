import tensorflow as tf
import numpy as np

import sys

from random import randint

import midi_io
import midi_coder
import console

from constants import *

duration = 400

def load_model(session, path):
    new_saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
    new_saver.restore(session, path)
    return tf.get_default_graph()

if __name__ == "__main__":
    data = midi_coder.encode_list(midi_io.read_midi('data/rach.mid'))

    sample_count, feat_len = data.shape
    start = randint(0, sample_count - WINDOW - 1)
    frames = data[start:start+WINDOW]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        graph = load_model(sess, OUTPUT_DIR + 'graph')

        for i in range(duration):
            seed = frames[i:i+WINDOW]

            input = np.reshape(seed, (1, WINDOW, feat_len))

            predict_op = graph.get_operation_by_name('prediction/prediction')
            predicted_tensor = predict_op.outputs[0]

            input_tensor = graph.get_operation_by_name('input/x-input').outputs[0]

            output = sess.run(predicted_tensor, feed_dict={input_tensor: input})
            note = np.argmax(output[0])
            next = np.reshape(midi_coder.encode(note), (1, feat_len))

            frames = np.concatenate((frames, next))
            console.progress(i, duration, midi_coder.note_name(note))
        print ""

        decoded = tf.reshape(tf.argmax(frames, axis = 1), (-1,))
        notes = sess.run(decoded)
        midi_io.write_notes(notes, OUTPUT_DIR + 'gen.mid')
