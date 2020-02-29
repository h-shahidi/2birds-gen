import os
import sys
import random

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from modules.encoder import Encoder
from modules.decoder import Decoder
from utils import padding_utils
from utils import rouge

BETA = 0.017

class ModelGraph(object):
    def __init__(self, word_vocab=None, POS_vocab=None, NER_vocab=None, flags=None, mode='ce_train'):
        # mode can have the following values:
        #  'ce_train',
        #  'rl_train',
        #  'evaluate',
        #  'evaluate_bleu',
        #  'decode'.
        # it is different from mode in decoder which can be
        # 'ce_train', 'loss', 'greedy' or 'sample'
        self.mode = mode

        # is_training controls whether to use dropout
        is_training = True if mode in ('ce_train') else False

        self.flags = flags
        self.word_vocab = word_vocab

        # create placeholders
        self.create_placeholders()

        # create encoder
        self.encoder = Encoder(
            flags, 
            self.passage_words,
            self.passage_POSs,
            self.passage_NERs,
            self.passage_lengths,
            self.answer_span,
            word_vocab=word_vocab, 
            POS_vocab=POS_vocab, 
            NER_vocab=NER_vocab)

        # encode the input instance
        self.encoder_dim, self.encoder_hidden_states, self.encoder_features, self.decoder_init_state = self.encoder.encode(is_training=is_training)

        max_passage_length = tf.shape(self.passage_words)[1]
        self.passage_mask = tf.sequence_mask(self.passage_lengths, max_passage_length, dtype=tf.float32)

        loss_weights = tf.sequence_mask(self.question_lengths, flags.max_question_len, dtype=tf.float32) # [batch_size, gen_steps]
        loss_weights_rl = tf.sequence_mask(self.question_lengths_rl, flags.max_question_len, dtype=tf.float32) # [batch_size, gen_steps]

        with tf.variable_scope("generator"):
            # create decoder
            self.decoder = Decoder(flags, word_vocab, self.rewards, is_training)

            if mode == 'decode':
                self.context_t_1 = tf.placeholder(tf.float32, [None, self.encoder_dim], name='context_t_1') # [batch_size, encoder_dim]
                self.coverage_t_1 = tf.placeholder(tf.float32, [None, None], name='coverage_t_1') # [batch_size, encoder_dim]
                self.word_t = tf.placeholder(tf.int32, [None], name='word_t') # [batch_size]

                (self.state_t, self.context_t, self.coverage_t, self.attn_dist_t, self.p_gen_t, self.ouput_t,
                    self.topk_log_probs, self.topk_ids, self.greedy_prediction, self.multinomial_prediction) = self.decoder.decode(
                        self.decoder_init_state, self.context_t_1, self.coverage_t_1, self.word_t, 
                        self.encoder_hidden_states, self.encoder_features, self.passage_words, self.passage_mask)
                # not buiding training op for this mode
                return

            elif mode == 'evaluate_bleu':
                _, _, self.greedy_words = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states, self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, loss_weights, mode='greedy')
                # not buiding training op for this mode
                return

            elif mode in ('ce_train', 'evaluate'):
                self.accu, self.loss, _ = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states, self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, loss_weights, mode='ce_train')
                if mode == 'evaluate':
                    # not buiding training op for evaluation
                    return

            elif mode == 'rl_train':
                _, self.loss, _ = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states,self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, loss_weights, mode='loss')

                tf.get_variable_scope().reuse_variables()

                _, _, self.greedy_words = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states,self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, None, mode='greedy')

            elif mode == 'rl_ce_train':

                self.accu, self.ce_loss, _ = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states, self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, loss_weights, mode='ce_train')
                
                tf.get_variable_scope().reuse_variables()

                _, self.rl_loss, _ = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states,self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs_rl, self.question_words_rl, loss_weights_rl, mode='loss')

                self.loss = BETA * self.ce_loss + self.rl_loss

                _, _, self.greedy_words = self.decoder.train(
                    self.encoder_dim, self.encoder_hidden_states,self.encoder_features,
                    self.passage_words, self.passage_mask, self.decoder_init_state,
                    self.decoder_inputs, self.question_words, None, mode='greedy')

        # defining optimizer and train op
        optimizer = tf.train.AdagradOptimizer(learning_rate=flags.learning_rate)

        tvars = tf.trainable_variables()
        total_parameters = 0
        for variable in tvars:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total number of parameters is equal: %s" % total_parameters)
        
        if flags.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + flags.lambda_l2 * l2_loss

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), flags.clip_value)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        with tf.control_dependencies([self.train_op]):
            self.train_op = ema.apply(tvars)
        with tf.variable_scope('backup_variables'):
            backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value()) 
                            for var in tvars]
        save_backup_vars_op = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(tvars, backup_vars)))
        with tf.control_dependencies([save_backup_vars_op]):
            self.ema_to_vars_op = tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in tvars))
        self.restore_backup_vars_op = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(tvars, backup_vars)))

    def create_placeholders(self):
        # build placeholder for input passage/article
        self.passage_lengths = tf.placeholder(tf.int32, [None], name='passage_lengths')
        self.passage_words = tf.placeholder(tf.int32, [None, None], name="passage_words") # [batch_size, passage_len]
        self.passage_POSs = tf.placeholder(tf.int32, [None, None], name="passage_POSs") # [batch_size, passage_len]
        self.passage_NERs = tf.placeholder(tf.int32, [None, None], name="passage_NERs") # [batch_size, passage_len]

        # build placeholder for answer
        self.answer_span = tf.placeholder(tf.float32, [None, None], name="answer_span")# [batch_size]

        # build placeholder for question
        self.decoder_inputs = tf.placeholder(tf.int32, [None, self.flags.max_question_len], name="decoder_inputs") # [batch_size, gen_steps]
        self.question_words = tf.placeholder(tf.int32, [None, self.flags.max_question_len], name="question_words") # [batch_size, gen_steps]
        self.question_lengths = tf.placeholder(tf.int32, [None], name="question_lengths") # [batch_size]

        self.decoder_inputs_rl = tf.placeholder(tf.int32, [None, self.flags.max_question_len], name="decoder_inputs_rl") # [batch_size, gen_steps]
        self.question_words_rl = tf.placeholder(tf.int32, [None, self.flags.max_question_len], name="question_words_rl") # [batch_size, gen_steps]
        self.question_lengths_rl = tf.placeholder(tf.int32, [None], name="question_lengths_rl") # [batch_size]

        # build placeholder for reinforcement learning
        self.rewards = tf.placeholder(tf.float32, [None], name="rewards")

    def run_greedy(self, sess, batch):
        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.decoder_inputs] = batch.decoder_inputs
        return sess.run(self.greedy_words, feed_dict)

    def ce_train(self, sess, batch, only_eval=False):
        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.decoder_inputs] = batch.decoder_inputs
        feed_dict[self.question_words] = batch.question_words
        feed_dict[self.question_lengths] = batch.question_lengths

        if only_eval:
            return sess.run([self.accu, self.loss], feed_dict)
        else:
            return sess.run([self.train_op, self.loss], feed_dict)[1]

    def rl_train(self, sess, batch):
        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.decoder_inputs] = batch.decoder_inputs

        greedy_outputs = sess.run(self.greedy_words, feed_dict)
        greedy_outputs = greedy_outputs.tolist()
        gold_output = batch.question_words.tolist()

        # baseline outputs by flipping coin
        flipp = 0.1
        baseline_outputs = np.copy(batch.question_words)
        for i in range(batch.question_words.shape[0]):
            seq_len = min(self.flags.max_question_len, batch.question_lengths[i]-1) # don't change stop token '</s>'
            for j in range(seq_len):
                if greedy_outputs[i][j] != 0 and random.random() < flipp:
                    baseline_outputs[i,j] = greedy_outputs[i][j]
        baseline_outputs = baseline_outputs.tolist()

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        rewards = []
        for i, (baseline_output, greedy_output) in enumerate(zip(baseline_outputs, greedy_outputs)):
            _, baseline_output_words = self.word_vocab.getLexical(baseline_output)
            greedy_output, greedy_output_words = self.word_vocab.getLexical(greedy_output)
            _, gold_ouput_words = self.word_vocab.getLexical(gold_output[i])

            rl_inputs.append([int(batch.decoder_inputs[i,0])] + greedy_output[:-1])
            rl_outputs.append(greedy_output)
            rl_input_lengths.append(len(greedy_output))

            baseline_output_words_list = baseline_output_words.split()
            greedy_output_words_list = greedy_output_words.split()
            gold_output_words_list = gold_ouput_words.split()

            if self.flags.reward_type == 'bleu':
                cc = SmoothingFunction()
                reward = sentence_bleu([gold_output_words_list], greedy_output_words_list, smoothing_function=cc.method3)
                baseline = sentence_bleu([gold_output_words_list], baseline_output_words_list, smoothing_function=cc.method3)
                rewards.append(reward - baseline)

            elif self.flags.reward_type == 'rouge':
                reward = rouge.rouge([gold_ouput_words], [greedy_output_words])["rouge_l/f_score"]
                baseline = rouge.rouge([gold_ouput_words], [baseline_output_words])["rouge_l/f_score"]
                rewards.append(reward - baseline)

            else:
                raise ValueError("Reward type is not bleu or rouge!")

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.flags.max_question_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.flags.max_question_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        #reward = rescale(reward)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.rewards] = rewards
        feed_dict[self.decoder_inputs] = rl_inputs
        feed_dict[self.question_words] = rl_outputs
        feed_dict[self.question_lengths] = rl_input_lengths

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def rl_ce_train(self, sess, batch):
        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.decoder_inputs] = batch.decoder_inputs

        # get greedy and gold outputs
        greedy_output = sess.run(self.greedy_words, feed_dict)
        greedy_output = greedy_output.tolist()
        gold_output = batch.question_words.tolist()

        # baseline outputs by flipping coin
        flipp = 0.1
        baseline_outputs = np.copy(batch.question_words)
        for i in range(batch.question_words.shape[0]):
            seq_len = min(self.flags.max_question_len, batch.question_lengths[i]-1) # don't change stop token '</s>'
            for j in range(seq_len):
                if greedy_output[i][j] != 0 and random.random() < flipp:
                    baseline_outputs[i,j] = greedy_output[i][j]
        baseline_outputs = baseline_outputs.tolist()

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        rewards = []
        for i, (baseline_output, greedy_output) in enumerate(zip(baseline_outputs, greedy_output)):
            _, baseline_output_words = self.word_vocab.getLexical(baseline_output)
            greedy_output, greedy_output_words = self.word_vocab.getLexical(greedy_output)
            _, gold_output_words = self.word_vocab.getLexical(gold_output[i])

            rl_inputs.append([int(batch.decoder_inputs[i,0])] + greedy_output[:-1])
            rl_outputs.append(greedy_output)
            rl_input_lengths.append(len(greedy_output))
            
            baseline_output_words_list = baseline_output_words.split()
            greedy_output_words_list = greedy_output_words.split()
            gold_output_words_list = gold_output_words.split()

            if self.flags.reward_type == 'bleu':
                cc = SmoothingFunction()
                reward = sentence_bleu([gold_output_words_list], greedy_output_words_list, smoothing_function=cc.method3)
                baseline = sentence_bleu([gold_output_words_list], baseline_output_words_list, smoothing_function=cc.method3)
                rewards.append(reward - baseline)

            elif self.flags.reward_type == 'rouge':
                reward = rouge.rouge([gold_output_words], [greedy_output_words])["rouge_l/f_score"]
                baseline = rouge.rouge([gold_output_words], [baseline_output_words])["rouge_l/f_score"]
                rewards.append(reward - baseline)

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.flags.max_question_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.flags.max_question_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
#        reward = rescale(reward)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, only_feed_dict=True)
        feed_dict[self.rewards] = rewards
        feed_dict[self.decoder_inputs_rl] = rl_inputs
        feed_dict[self.question_words_rl] = rl_outputs
        feed_dict[self.question_lengths_rl] = rl_input_lengths

        feed_dict[self.decoder_inputs] = batch.decoder_inputs
        feed_dict[self.question_words] = batch.question_words
        feed_dict[self.question_lengths] = batch.question_lengths

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def run_encoder(self, sess, batch, only_feed_dict=False):
        feed_dict = {}
        feed_dict[self.passage_lengths] = batch.sent1_length
        if self.flags.with_word: 
            feed_dict[self.passage_words] = batch.sent1_word
        if self.flags.with_POS: 
            feed_dict[self.passage_POSs] = batch.sent1_POS
        if self.flags.with_NER: 
            feed_dict[self.passage_NERs] = batch.sent1_NER
        if self.flags.with_answer_span:
            feed_dict[self.answer_span] = batch.answer_span

        if only_feed_dict:
            return feed_dict

        return sess.run([self.encoder_hidden_states, self.decoder_init_state, self.encoder_features, 
                         self.passage_words, self.passage_mask], feed_dict)


def redistribution(idx, total):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward):
    batch_size = reward.shape[0]
    result = np.zeros((batch_size))

    reward_dict = {}
    for r in reward:
        reward_dict[r] = r

    idx = 1
    for r in reward_dict:
        reward_dict[r] = redistribution(idx, len(reward))
        idx += 1

    for i in range(batch_size):
        result[i] = reward_dict[reward[i]]

    return result