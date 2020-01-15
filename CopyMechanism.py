import tensorflow as tf
import pickle

class copy_mech():
    def __init__(self, emb_size, hidden_size):
        self.hidden_size = hidden_size
        self.params = {}

        with tf.variable_scope('copy_mech'):
            self.w = tf.get_variable("w", [hidden_size*5, 1])
            self.b = tf.get_variable("b", [1], initializer=tf.zeros_initializer([1]))

        self.params.update({'w':self.w, 'b':self.b})

    def __call__(self, input_t_trans, s_nt, c_t, w_t, vocab_dist_t, input_encoder, encoder_len):
        x = tf.concat([input_t_trans, s_nt[0], s_nt[1], c_t], axis=1)
        p_gen = tf.sigmoid(tf.nn.xw_plus_b(x, self.w, self.b))

        input_shape = tf.shape(vocab_dist_t)
        batch_size = input_shape[0]
        vsize = input_shape[1]
        passage_length = tf.shape(input_encoder)[1]
        w_t = tf.transpose(tf.squeeze(w_t, [2]), [1,0])
        with tf.variable_scope('final_distribution'):
            vocab_dist = p_gen * vocab_dist_t
            with tf.control_dependencies([tf.assert_equal(tf.shape(w_t), tf.shape(input_encoder))]):
                attn_dist = (1.0-p_gen) * w_t

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for phrases
            extended_vsize = vsize
            # match attn_dist[batch_size, passage_length] to sparse one-hot representation [batch_size, passage_length, extended_vsize]
            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, axis=1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, passage_length]) # shape (batch_size, passage_length)
            step_nums = tf.range(0, limit=passage_length) # [passage_length]
            step_nums = tf.expand_dims(step_nums, axis=0) # shape (1, passage_length)
            step_nums = tf.tile(step_nums, [batch_size, 1]) # shape (batch_size, passage_length)
            indices = tf.stack((batch_nums, step_nums, input_encoder), axis=2) # shape (batch_size, passage_length, 3)
            indices = tf.reshape(indices, [-1, 3]) #[batch_size * passage_length, 3]
            indices = tf.cast(indices, tf.int64)

            shape = [batch_size, passage_length, extended_vsize]
            shape = tf.cast(shape, tf.int64)

            attn_dist = tf.reshape(attn_dist, shape=[-1]) # [batch_size*passage_length]
            one_hot_spare_rep = tf.SparseTensor(indices=indices, values=attn_dist, dense_shape=shape) # [batch_size, passage_length, extended_vsize]

            if encoder_len is not None:
                passage_mask = tf.sequence_mask(encoder_len, passage_length, dtype=tf.float32)
                passage_mask = tf.expand_dims(passage_mask, axis=-1)
                one_hot_spare_rep = one_hot_spare_rep * passage_mask

            one_hot_spare_rep = tf.sparse_reduce_sum(one_hot_spare_rep, axis=1) # [batch_size, extended_vsize]
            one_hot_spare_rep.set_shape([None, None])
            vocab_dist = tf.add(vocab_dist, one_hot_spare_rep)

        return vocab_dist

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
        
