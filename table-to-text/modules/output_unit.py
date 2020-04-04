# This code is based on https://github.com/tyliupku/wiki2bio.
# We would like to thank the authors for sharing their code base.

import tensorflow as tf
import pickle

class OutputUnit(object):
    def __init__(self, input_size, output_size, scope_name):
        self.input_size = input_size
        self.output_size = output_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [input_size, output_size])
            self.b = tf.get_variable('b', [output_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.Wo = tf.get_variable('Wo', [3*input_size, input_size])
            self.bo = tf.get_variable('bo', [input_size])

        self.params.update({'W': self.W, 'b': self.b, 'Wo': self.Wo, 'bo': self.bo})

    def __call__(self, c_t, o_t, finished = None):
        x = tf.nn.xw_plus_b(tf.concat([c_t, o_t], -1), self.Wo, self.bo)
        out = tf.nn.xw_plus_b(x, self.W, self.b)
        out = tf.nn.softmax(out)
        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out

    def save(self, path):
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        param_values = pickle.load(open(path, 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])
