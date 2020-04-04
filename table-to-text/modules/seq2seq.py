# This code is based on https://github.com/tyliupku/wiki2bio.
# We would like to thank the authors for sharing their code base.

import tensorflow as tf
import pickle
import numpy as np
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

from modules.attention import AttentionWrapper
from modules.lstm_unit import LSTMUnit
from modules.output_unit import OutputUnit
from modules.copy_mechanism import copy_mech
import utils.rouge

BETA = 0.017

class SeqUnit(object):
    def __init__(self, FLAGS, scope_name, is_training, start_token=2, stop_token=2, max_length=150):
        '''
        batch_size, hidden_size, emb_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        source_vocab, target_vocab, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        '''
        self.batch_size = FLAGS.batch_size
        self.hidden_size = FLAGS.hidden_size
        self.emb_size = FLAGS.emb_size
        self.field_size = FLAGS.field_size
        self.pos_size = FLAGS.pos_size
        self.uni_size = FLAGS.emb_size + FLAGS.field_size + 2 * FLAGS.pos_size
        self.source_vocab = FLAGS.source_vocab
        self.target_vocab = FLAGS.target_vocab
        self.field_vocab = FLAGS.field_vocab
        self.position_vocab = FLAGS.position_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.n_iter = 0
        self.baseline = 0

        self.units = {}
        self.params = {}

        self.encoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_field = tf.placeholder(tf.int32, [None, None])
        self.encoder_pos = tf.placeholder(tf.int32, [None, None])
        self.encoder_rpos = tf.placeholder(tf.int32, [None, None])
        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, None])
        self.decoder_output_gold = tf.placeholder(tf.int32, [None, None])
        self.rewards = tf.placeholder(tf.float32, [None])
        self.enc_mask = tf.sign(tf.to_float(self.encoder_pos))
        with tf.variable_scope(scope_name):
            self.dec_lstm = LSTMUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
            self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')

        self.copy_mech = copy_mech(self.emb_size, self.hidden_size)

        with tf.variable_scope("context_concat"):
            self.Wc = tf.get_variable('Wc', [2 * self.hidden_size + self.emb_size, self.hidden_size])
            self.bc = tf.get_variable('bc', [self.hidden_size])

        with tf.variable_scope("decoder_init"):
            self.W_init_h = tf.get_variable('W_init_h', [2 * self.hidden_size, self.hidden_size])
            self.b_init_h = tf.get_variable('b_init_h', [self.hidden_size])
            self.W_init_c = tf.get_variable('W_init_c', [2 * self.hidden_size, self.hidden_size])
            self.b_init_c = tf.get_variable('b_init_c', [self.hidden_size])

        self.params.update({'Wc': self.Wc, 'bc': self.bc, 
                            'W_init_h': self.W_init_h, 'b_init_h': self.b_init_h, 'W_init_c': self.W_init_c, 'b_init_c': self.b_init_c})

        self.units.update({'decoder_lstm': self.dec_lstm, 
                           'decoder_output': self.dec_out, 
                           'copy_mech': self.copy_mech})

        # ======================================== embeddings ======================================== #
        with tf.device('/cpu:0'):
            with tf.variable_scope(scope_name):
                self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
                self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
                self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
                #if is_training:
                #    self.decoder_embed = tf.nn.dropout(self.decoder_embed, 0.7)
                #    self.encoder_embed = tf.nn.dropout(self.encoder_embed, 0.7)
                self.fembedding = tf.get_variable('fembedding', [self.field_vocab, self.field_size])
                self.field_embed = tf.nn.embedding_lookup(self.fembedding, self.encoder_field)
                self.encoder_embed = tf.concat([self.encoder_embed, self.field_embed], 2)
                
                self.pembedding = tf.get_variable('pembedding', [self.position_vocab, self.pos_size])
                self.rembedding = tf.get_variable('rembedding', [self.position_vocab, self.pos_size])
                self.pos_embed = tf.nn.embedding_lookup(self.pembedding, self.encoder_pos)
                self.rpos_embed = tf.nn.embedding_lookup(self.rembedding, self.encoder_rpos)
                self.encoder_embed = tf.concat([self.encoder_embed, self.pos_embed, self.rpos_embed], 2)

        self.params.update({'fembedding': self.fembedding})
        self.params.update({'pembedding': self.pembedding})
        self.params.update({'rembedding': self.rembedding})
        self.params.update({'embedding': self.embedding})

        # ======================================== encoder ======================================== #
        with tf.variable_scope('bi_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            #if is_training:
            #    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob = 0.7)
            #    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob = 0.7)

            (en_outputs_fw, en_outputs_bw), (en_state_fw, en_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                         self.encoder_embed,
                                                                                                         sequence_length=self.encoder_len,
                                                                                                         dtype=tf.float32)
            en_outputs = tf.concat([en_outputs_fw, en_outputs_bw], axis=-1)
            en_state_h = tf.concat([en_state_fw.h, en_state_bw.h], axis=-1)
            en_state_c = tf.concat([en_state_fw.c, en_state_bw.c], axis=-1)
            new_en_state_h = tf.nn.tanh(tf.nn.xw_plus_b(en_state_h, self.W_init_h, self.b_init_h))
            new_en_state_c = tf.nn.tanh(tf.nn.xw_plus_b(en_state_c, self.W_init_c, self.b_init_c))
            en_state = (new_en_state_h, new_en_state_c)

            bi_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bi_encoder')
            self.params.update({bi_encoder_var.name:bi_encoder_var for bi_encoder_var in bi_encoder_vars})

        # ======================================== decoder ======================================== #

        with tf.variable_scope(scope_name):
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs, self.encoder_len, "attention")
            self.units.update({'attention': self.att_layer})

        # decoder for training
        de_outputs, de_state = self.decoder_train(en_state, self.decoder_embed, self.decoder_len)
        # decoder for testing
        g_outputs, self.atts = self.decoder_test(en_state)
        self.g_tokens = tf.arg_max(g_outputs, 2)

        mask = tf.sign(tf.to_float(self.decoder_output))
        if FLAGS.loss == 'ce':
            losses = self.CE_loss(de_outputs, self.decoder_output, mask)
            self.mean_loss = tf.reduce_mean(losses)
        else:
            losses = self.CE_loss(g_outputs, self.decoder_output, mask)
            losses *= self.rewards
            losses += BETA * self.CE_loss(de_outputs, self.decoder_output_gold, tf.sign(tf.to_float(self.decoder_output_gold)))
            self.mean_loss = tf.reduce_mean(losses)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        opt_op = optimizer.apply_gradients(zip(grads, tvars))
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.ema = ema
        with tf.control_dependencies([opt_op]):
            self.train_op = ema.apply(tvars)
        with tf.variable_scope('backup_variables'):
            backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value()) 
                           for var in tvars]
        save_backup_vars_op = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(tvars, backup_vars)))
        with tf.control_dependencies([save_backup_vars_op]):
            self.ema_to_vars_op = tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in tvars))
        self.restore_backup_vars_op = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(tvars, backup_vars)))

    def decoder_train(self, initial_state, inputs, inputs_len):
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]
        encoder_len = tf.shape(self.encoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        c0 = tf.zeros([batch_size, 2*self.hidden_size], dtype=tf.float32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, c_t, s_t, emit_ta, finished):
            input_t = tf.concat([x_t, c_t], axis=-1)
            input_t_trans = tf.nn.xw_plus_b(input_t, self.Wc, self.bc)
            o_t, s_nt = self.dec_lstm(input_t_trans, s_t, finished)
            c_nt, w_t = self.att_layer(s_nt)
            vocab_dist_t = self.dec_out(c_nt, o_t, finished)
            vocab_dist_t = self.copy_mech(input_t_trans, s_nt, c_nt, w_t, vocab_dist_t, self.encoder_input, self.encoder_len)
            emit_ta = emit_ta.write(t, vocab_dist_t)
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t))
            return t+1, x_nt, c_nt, s_nt, emit_ta, finished

        _, _, _, state, emit_ta,  _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, c0, h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def decoder_test(self, initial_state):
        batch_size = tf.shape(self.encoder_input)[0]
        encoder_len = tf.shape(self.encoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        c0 = tf.zeros([batch_size, 2*self.hidden_size], dtype=tf.float32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        att_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, c_t, s_t, emit_ta, att_ta, finished):
            input_t = tf.concat([x_t, c_t], axis=-1)
            input_t_trans = tf.nn.xw_plus_b(input_t, self.Wc, self.bc)
            o_t, s_nt = self.dec_lstm(input_t_trans, s_t, finished)
            c_nt, w_t = self.att_layer(s_nt)
            vocab_dist_t = self.dec_out(c_nt, o_t, finished)
            vocab_dist_t = self.copy_mech(input_t_trans, s_nt, c_nt, w_t, vocab_dist_t, self.encoder_input, self.encoder_len)
            emit_ta = emit_ta.write(t, vocab_dist_t)
            att_ta = att_ta.write(t, w_t)
            next_token = tf.arg_max(vocab_dist_t, 1)
            x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, c_nt, s_nt, emit_ta, att_ta, finished

        _, _, _, state, emit_ta, att_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, _6, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, c0, h0, emit_ta, att_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        atts = att_ta.stack()
        return outputs, atts

    def ce_train(self, x, sess):
        loss,  _ = sess.run([self.mean_loss, self.train_op],
                           {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
                            self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
                            self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
                            self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']})
        return loss

    def rl_train(self, x, sess, vocab):
        def redistribution(idx, total, min_v):
            idx = (idx + 0.0) / (total + 0.0) * 16.0
            return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))

        def rescale(reward):
            reward_dim_0 = reward.shape[0]
            ret = np.zeros((reward_dim_0))
        
            rescalar = {}
            for r in reward:
                rescalar[r] = r

            idx = 1
            min_s = 1.0
            max_s = 0.0
            for r in rescalar:
                rescalar[r] = redistribution(idx, len(reward), min_s)
                idx += 1
            
            for i in range(reward_dim_0):
                ret[i] = rescalar[reward[i]]

            return ret

        predictions, _ = self.generate(x, sess)

        rewards = []
        decoder_len = []
        decoder_input = []
        decoder_output = []
        for i,pred in enumerate(predictions):
            if 2 in pred:
                eos_index = np.where(pred==2)[0][0]
                pred = pred[:eos_index] if pred[0] != 2 else [2]
            decoder_len.append(len(pred))
            decoder_input.append(pred)
            decoder_output.append(pred)
            toks = []
            for tok in pred:
                toks.append(vocab.id2word(tok))                
            rewards.append(sentence_bleu([x['summary_toks'][i]], toks, smoothing_function=cc.method3))
            #rewards.append(sentence_gleu([x['summary_toks'][i]], toks))
            #ref = [item.decode("utf-8") for item in x['summary_toks'][i]]
            #can = [item.decode("utf-8") for item in toks]
            #rewards.append(rouge.rouge([" ".join(ref)], [" ".join(can)])["rouge_l/f_score"])

        rewards = np.array(rewards, dtype=np.float32)
        rewards = rescale(rewards)
        rewards = rewards - self.baseline
        self.baseline = (self.baseline * self.n_iter + np.mean(rewards)) / (self.n_iter + 1)
        self.n_iter += 1
        max_len = max(decoder_len)
        decoder_input_padded = []
        decoder_output_padded = []
        for i in range(len(decoder_input)):
            decoder_input_padded.append(np.concatenate((decoder_input[i], [0] * (predictions.shape[1] - decoder_len[i]))))
            if decoder_len[i] <= self.max_length:
                decoder_output_padded.append(np.concatenate((decoder_output[i], [2] + [0] * (predictions.shape[1]-1-decoder_len[i]))))
            else:
                decoder_output_padded.append(decoder_output[i])

        loss,  _ = sess.run([self.mean_loss, self.train_op],
                           {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'],
                            self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'],
                            self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
                            self.decoder_len: x['dec_len'], self.decoder_output: decoder_output_padded, self.rewards: rewards,
                            self.decoder_output_gold: x['dec_out']})
        return loss             

    def generate(self, x, sess):
        predictions, atts = sess.run([self.g_tokens, self.atts],
                               {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'], 
                                self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
                                self.encoder_rpos: x['enc_rpos']})
        return predictions, atts

    def CE_loss(self, word_probs, answers, loss_weights):
        '''
        word_probs: [batch_size, max_dec_steps, vocab]
        answers: [batch_size, max_dec_steps]
        loss_weigts: [batch_size, max_dec_steps]
        '''
        def _clip_and_normalize(word_probs, epsilon):
            '''
            word_probs: 1D tensor of [vsize]
            '''
            word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
            return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True) # scale preds so that the class probas of each sample sum to 1
        
        input_shape = tf.shape(word_probs)
        vsize = input_shape[2]
   
        epsilon = 1.0e-6
        word_probs = _clip_and_normalize(word_probs, epsilon)
   
        one_hot_spare_rep = tf.one_hot(answers, vsize)
        xent = -tf.reduce_sum(one_hot_spare_rep * tf.log(word_probs), axis=-1) # [batch_size, max_dec_steps]
        if loss_weights != None:
            xent = xent * loss_weights
        xent = tf.reduce_sum(xent, axis=-1)
        return xent #[batch_size]

    def save(self, path):
        for u in self.units:
            self.units[u].save(path+u+".pkl")
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path+self.scope_name+".pkl", 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        for u in self.units:
            self.units[u].load(path+u+".pkl")
        param_values = pickle.load(open(path+self.scope_name+".pkl", 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])

