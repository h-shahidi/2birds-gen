import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import nn_ops


class Encoder(object):
    def __init__(self,
                 flags, 
                 passage_words,
                 passage_POSs,
                 passage_NERs,
                 passage_lengths,
                 answer_span,
                 word_vocab=None, 
                 POS_vocab=None, 
                 NER_vocab=None):
                 
        self.flags = flags
        self.word_vocab = word_vocab
        self.POS_vocab = POS_vocab
        self.NER_vocab = NER_vocab
        self.passage_lengths = passage_lengths         #tf.placeholder(tf.int32, [None])
        self.passage_words = passage_words             #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
        self.passage_POSs = passage_POSs               #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]=
        self.passage_NERs = passage_NERs               #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
        self.answer_span = answer_span

    def calculate_encoder_features(self, encoder_states, encoder_dim):
        input_shape = tf.shape(encoder_states)
        batch_size = input_shape[0]
        passage_len = input_shape[1]

        with tf.variable_scope("encoder_features"):
            encoder_features = tf.expand_dims(encoder_states, axis=2) # now is shape [batch_size, passage_len, 1, encoder_dim]
            W_h = tf.get_variable("W_h", [1, 1, encoder_dim, self.flags.attention_vec_size])
            encoder_features = nn_ops.conv2d(encoder_features, W_h, [1, 1, 1, 1], "SAME") # [batch_size, passage_len, 1, attention_vec_size]
            encoder_features = tf.reshape(encoder_features, [batch_size, passage_len, self.flags.attention_vec_size])
        return encoder_features

    def encode(self, is_training=True):
        passage_representations = []
        if self.flags.with_word and self.word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if self.flags.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'        
            with tf.variable_scope("embedding"), tf.device(cur_device):
                self.word_embedding = tf.get_variable(
                    "word_embedding", trainable=word_vec_trainable,
                    initializer=tf.constant(self.word_vocab.word_vecs), dtype=tf.float32)
            passage_word_representations = tf.nn.embedding_lookup(self.word_embedding, self.passage_words) # [batch_size, passage_len, word_dim]
            passage_representations.append(passage_word_representations)

        if self.flags.with_POS and self.POS_vocab is not None:
            self.POS_embedding = tf.get_variable("POS_embedding", initializer=tf.constant(self.POS_vocab.word_vecs), dtype=tf.float32)
            passage_POS_representation = tf.nn.embedding_lookup(self.POS_embedding, self.passage_POSs) # [batch_size, passage_len, POS_dim]
            passage_representations.append(passage_POS_representation)

        if self.flags.with_NER and self.NER_vocab is not None:
            self.NER_embedding = tf.get_variable("NER_embedding", initializer=tf.constant(self.NER_vocab.word_vecs), dtype=tf.float32)
            passage_NER_representation = tf.nn.embedding_lookup(self.NER_embedding, self.passage_NERs) # [batch_size, passage_len, NER_dim]
            passage_representations.append(passage_NER_representation)

        if self.flags.with_answer_span:
            passage_representations.append(tf.expand_dims(self.answer_span, axis=-1))

        passage_representations = tf.concat(passage_representations, 2) # [batch_size, passage_len, dim]

        if is_training:
            passage_representations = tf.nn.dropout(passage_representations, (1 - self.flags.dropout_rate))
        else:
            passage_representations = tf.multiply(passage_representations, (1 - self.flags.dropout_rate))

        passage_len = tf.shape(self.passage_words)[1]
        passage_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]

        with tf.variable_scope('encoding'):
            cur_passage_representation = passage_representations
            for i in xrange(self.flags.encoder_layer_num):
                with tf.variable_scope('layer-{}'.format(i)):
                    encoder_lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.flags.encoder_hidden_size)
                    encoder_lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.flags.encoder_hidden_size)
                    if is_training:
                        encoder_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_lstm_cell_fw, output_keep_prob=(1 - self.flags.dropout_rate))
                        encoder_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(encoder_lstm_cell_bw, output_keep_prob=(1 - self.flags.dropout_rate))

                    ((passage_representations_fw, passage_representations_bw), (passage_last_fw, passage_last_bw)) = tf.nn.bidirectional_dynamic_rnn(
                        encoder_lstm_cell_fw, encoder_lstm_cell_bw, cur_passage_representation, dtype=tf.float32,
                        sequence_length=self.passage_lengths) # [batch_size, passage_len, encoder_hidden_size]

                    cur_passage_representation = tf.concat([passage_representations_fw, passage_representations_bw], 2)

        encoder_size = self.flags.encoder_hidden_size * 2
        encoder_hidden_states = cur_passage_representation 
        encoder_hidden_states = encoder_hidden_states * tf.expand_dims(passage_mask, axis=-1)

        encoder_features = self.calculate_encoder_features(encoder_hidden_states, encoder_size)

        # initial state for the LSTM decoder
        with tf.variable_scope('decoder_initial_state'):
            w_c = tf.get_variable('w_c', [2*self.flags.encoder_hidden_size, self.flags.decoder_hidden_size], dtype=tf.float32)
            w_h = tf.get_variable('w_h', [2*self.flags.encoder_hidden_size, self.flags.decoder_hidden_size], dtype=tf.float32)
            bias_c = tf.get_variable('bias_c', [self.flags.decoder_hidden_size], dtype=tf.float32)
            bias_h = tf.get_variable('bias_h', [self.flags.decoder_hidden_size], dtype=tf.float32)

            old_c = tf.concat([passage_last_fw.c, passage_last_bw.c], axis=-1)
            old_h = tf.concat([passage_last_fw.h, passage_last_bw.h], axis=-1)
            new_c = tf.nn.tanh(tf.matmul(old_c, w_c) + bias_c)
            new_h = tf.nn.tanh(tf.matmul(old_h, w_h) + bias_h)

            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        return (encoder_size, encoder_hidden_states, encoder_features, decoder_initial_state)