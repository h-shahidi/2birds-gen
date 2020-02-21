import tensorflow as tf
import pickle

class AttentionWrapper(object):
    def __init__(self, hidden_size, input_size, hs, enc_len, scope_name):
        self.hs = tf.transpose(hs, [1,0,2])
        self.enc_len = enc_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope('attention'):
            self.v = tf.get_variable("v", [hidden_size])
            self.Wh = tf.get_variable('Wh', [2*input_size, hidden_size])
            self.bh = tf.get_variable('bh', [hidden_size])           
            self.Ws = tf.get_variable('Ws', [2*input_size, hidden_size])
            self.bs = tf.get_variable('bs', [hidden_size])

        self.params.update({'Wh': self.Wh, 'Ws': self.Ws,
                            'bh': self.bh, 'bs': self.bs, 'v': self.v })
        
        hs2d = tf.reshape(self.hs, [-1, 2*input_size])
        phi_hs2d = tf.nn.xw_plus_b(hs2d, self.Wh, self.bh)
        self.phi_hs = tf.reshape(phi_hs2d, [tf.shape(self.hs)[0], tf.shape(self.hs)[1], hidden_size])

    def __call__(self, x, finished = None):
        # context: [batch_size, hidden_size]
        # weights: [passage_len, batch_size, 1]
        x_concat = tf.concat([x[0], x[1]], axis=-1)
        state_features = tf.nn.xw_plus_b(x_concat, self.Ws, self.bs)
        all_features = self.phi_hs + state_features
        v = tf.expand_dims(tf.expand_dims(self.v, axis=0), axis=0)
        e = tf.reduce_sum(v * tf.tanh(all_features), axis=-1, keep_dims=True) 
        attn_dist = tf.nn.softmax(e, dim=0) 
        attn_dist *= tf.expand_dims(tf.transpose(tf.sequence_mask(self.enc_len, tf.shape(self.hs)[0], dtype=tf.float32), [1,0]), -1)
        out = tf.reduce_sum(self.hs * attn_dist, axis=0)

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out, attn_dist

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
