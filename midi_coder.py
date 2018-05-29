import numpy as np
import tensorflow as tf


MIDI_COUNT = 128


def encode(note):
    """
        Returns the one-hot encoding for the provided midi note.
    """
    onehot = np.zeros([MIDI_COUNT], dtype=np.float32)
    onehot[note] = 1.0
    return onehot


def encode_list(notes):
    return np.array(map(encode, notes))


def encode_tensor(notes):
    return np.array(tf.map_fn(encode, notes))


def decode(onehot):
    """
        Returns the midi note for the provided one-hot encoding.
    """
    return np.argmax(onehot)


def decode_list(notes):
    return np.array(map(decode, notes))


def decode_tensor(notes):
    def tf_decode(onehot):
        return tf.argmax(onehot)
    return tf.map_fn(tf_decode, notes)


def note_name(midi_value):
    """
        Returns the note name for the provided midi value.
    """
    octave = midi_value / 12 - 2
    names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    note_name = names[midi_value % 12]
    return '{0}{1}'.format(note_name, octave)

def classes():
    return map(note_name, range(MIDI_COUNT))
