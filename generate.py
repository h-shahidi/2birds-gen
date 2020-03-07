from __future__ import print_function
import argparse
import os
import sys
import time
import string 
import re

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

from modules.data import read_data_split_1, read_data_split_2, QGDataLoader
from modules.model import ModelGraph
from utils.vocab_utils import Vocab
from utils import config_utils

tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


class Hypothesis(object):
    def __init__(self, tokens, log_ps, state, context_vector, coverage_vector=None, attn_dist=None):
        self.tokens = tokens # store all tokens
        self.log_probs = log_ps # store log_probs for each time-step
        self.state = state
        self.context_vector = context_vector
        self.coverage_vector = coverage_vector
        self.attn_dist= attn_dist

    def extend(self, token, log_prob, state, context_vector, coverage_vector=None, attn_dist=None):
        return Hypothesis(
            self.tokens + [token], 
            self.log_probs + [log_prob], 
            state,
            context_vector, 
            coverage_vector=coverage_vector, 
            attn_dist = self.attn_dist + [attn_dist])

    def latest_token(self):
        return self.tokens[-1]

    def avg_log_prob(self):
        return np.sum(self.log_probs[1:])/ ((len(self.tokens)-1)**1.75)

    def probs2string(self):
        out_string = ""
        for prob in self.log_probs:
            out_string += " %.4f" % prob
        return out_string.strip()

    def idx_seq_to_string(self, passage, vocab):
        all_words = []
        for i, idx in enumerate(self.tokens):
            cur_word = vocab.getWord(idx)
            if  idx == vocab.vocab_size:
                j = 0
                passage_tokens = re.split('\\s+', passage.tokText)
                while j < len(passage_tokens):
                    max_index = np.argmax(self.attn_dist[i])
                    if passage.answer_span[max_index] == 1:
                        self.attn_dist[i][max_index] = -np.inf
                    else:
                        candidate_word = passage_tokens[max_index]
                        if candidate_word in stopwords.words('english'):
                            self.attn_dist[i][max_index] = -np.inf
                        else:
                            cur_word = candidate_word
                            break
                    j += 1
            all_words.append(cur_word)
        return " ".join(all_words[1:-1])


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)


def beam_search(sess, model, vocab, batch, flags):
    # Run encoder
    (passage_representations, decoder_init_state, encoder_features, passage_words, passage_mask) = model.run_encoder(sess, batch)
    # passage_representations: [1, passage_len, encoder_dim]
    # decoder_init_state: a tupel of [1, gen_dim]
    # encoder_features: [1, passage_len, attention_vec_size]
    # passage_words: [1, passage_len]
    # passage_mask: [1, passage_len]

    sent_stop_id = vocab.getIndex('</s>')

    # Initialize this first hypothesis
    context_t = np.zeros([model.encoder_dim]) # [encoder_dim]
    coverage_t = np.zeros((passage_representations.shape[1])) # [passage_len]
    attn_dist = np.zeros((passage_representations.shape[1]))
    
    hyps = []
    hyps.append(Hypothesis(
        [batch.decoder_inputs[0][0]], 
        [0.0], 
        decoder_init_state, 
        context_t, 
        coverage_vector=coverage_t, 
        attn_dist=[attn_dist]))

    # beam search decoding
    results = [] # this will contain finished hypotheses (those that have emitted the </s> token)
    steps = 0
    while steps < flags.max_question_len and len(results) < flags.beam_size:
        cur_size = len(hyps) # current number of hypothesis in the beam
        cur_passage_representations = np.tile(passage_representations, (cur_size, 1, 1))
        cur_encoder_features = np.tile(encoder_features, (cur_size, 1, 1)) # [batch_size,passage_len, flags.attention_vec_size]
        cur_passage_words = np.tile(passage_words, (cur_size, 1)) # [batch_size, passage_len]
        cur_passage_mask = np.tile(passage_mask, (cur_size, 1)) # [batch_size, passage_len]
        cur_state_t_1 = [] # [2, gen_steps]
        cur_context_t_1 = [] # [batch_size, encoder_dim]
        cur_coverage_t_1 = [] # [batch_size, passage_len]
        cur_word_t = [] # [batch_size]
        for h in hyps:
            cur_state_t_1.append(h.state)
            cur_context_t_1.append(h.context_vector)
            cur_word_t.append(h.latest_token())
            cur_coverage_t_1.append(h.coverage_vector)
        cur_context_t_1 = np.stack(cur_context_t_1, axis=0)
        cur_coverage_t_1 = np.stack(cur_coverage_t_1, axis=0)
        cur_word_t = np.array(cur_word_t)

        cell_states = [state.c for state in cur_state_t_1]
        hidden_states = [state.h for state in cur_state_t_1]
        new_c = np.concatenate(cell_states, axis=0)
        new_h = np.concatenate(hidden_states, axis=0)
        new_dec_init_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        feed_dict = {}
        feed_dict[model.decoder_init_state] = new_dec_init_state
        feed_dict[model.context_t_1] = cur_context_t_1
        feed_dict[model.word_t] = cur_word_t
        feed_dict[model.encoder_hidden_states] = cur_passage_representations
        feed_dict[model.encoder_features] = cur_encoder_features
        feed_dict[model.passage_words] = cur_passage_words
        feed_dict[model.passage_mask] = cur_passage_mask
        feed_dict[model.coverage_t_1] = cur_coverage_t_1
        if flags.with_answer_span:
            feed_dict[model.answer_span] = np.tile(batch.answer_span, (cur_size, 1))

        (state_t, context_t, attn_dist_t, coverage_t, topk_log_probs, topk_ids) = sess.run(
            [model.state_t, model.context_t, model.attn_dist_t,
            model.coverage_t, model.topk_log_probs, model.topk_ids], feed_dict)

        new_states = [tf.nn.rnn_cell.LSTMStateTuple(state_t.c[i:i+1, :], state_t.h[i:i+1, :]) for i in xrange(cur_size)]

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        for i in xrange(cur_size):
            h = hyps[i]
            cur_state = new_states[i]
            cur_context = context_t[i]
            cur_coverage = coverage_t[i]
            cur_attn_dist = attn_dist_t[i]
            for j in xrange(flags.beam_size):
                cur_tok = topk_ids[i, j]
                cur_tok_log_prob = topk_log_probs[i, j]
                new_hyp = h.extend(
                    cur_tok, 
                    cur_tok_log_prob, 
                    cur_state, cur_context, 
                    coverage_vector=cur_coverage, 
                    attn_dist=cur_attn_dist)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        # hyps will contain hypotheses for the next step
        hyps = []
        for h in sort_hyps(all_hyps):
            # If this hypothesis is sufficiently long, put in results. Otherwise discard.
            if h.latest_token() == sent_stop_id:
                if steps >= flags.min_question_len:
                    results.append(h)
            # hasn't reached stop token, so continue to extend this hypothesis
            else:
                hyps.append(h)
            if len(hyps) == flags.beam_size or len(results) == flags.beam_size:
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps
    # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    if len(results)==0:
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--out_path', type=str, help='The path to the output file.')

    args = parser.parse_args()

    model_prefix = args.model_prefix
    out_path = args.out_path

    # load the configuration file
    print('Loading configurations from ' + model_prefix + "config.json")
    FLAGS = config_utils.load_config(model_prefix + "config.json")

    # load vocabs
    print('Loading vocabs.')
    word_vocab = POS_vocab = NER_vocab = None
    if FLAGS.with_word:
        word_vocab = Vocab(embedding_path=FLAGS.word_vec_path)
        print('word_vocab: {}'.format(word_vocab.vocab_size))
    if FLAGS.with_POS:
        POS_vocab = Vocab(embedding_path=os.path.join(model_prefix, "POS_vocab"))
        print('POS_vocab: {}'.format(POS_vocab.vocab_size))
    if FLAGS.with_NER:
        NER_vocab = Vocab(embedding_path=os.path.join(model_prefix, "NER_vocab"))
        NER_vocab.word_vecs = NER_vocab.word_vecs[:-1,:]
        print('NER_vocab: {}'.format(NER_vocab.vocab_size))

    print('Loading test set.')
    if FLAGS.data_split == 1:
        testset, _ = read_data_split_1(FLAGS.s1_test_path, isLower=FLAGS.isLower)
    else:
        testset, _ = read_data_split_2(FLAGS.s2_test_path, isLower=FLAGS.isLower)
    print('Number of samples: {}'.format(len(testset)))

    batch_size = 1
    test_data_loader = QGDataLoader(
        testset, word_vocab, POS_vocab, NER_vocab, flags=FLAGS,
        isShuffle=False, isLoop=False, isSort=True, batch_size=batch_size)
    print('Number of instances in test data: {}'.format(test_data_loader.get_num_instance()))
    print('Number of batches in test data: {}'.format(test_data_loader.get_num_batch()))

    best_path = model_prefix + "best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, flags=FLAGS, mode="decode")

        # remove word _embedding
        _vars = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            _vars[var.name.split(":")[0]] = var
        saver = tf.train.Saver(_vars)

        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        total = 0
        correct = 0

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        ref_outfile = open(os.path.join(out_path, "output.ref"), 'wt')
        pred_outfile = open(os.path.join(out_path, "output.pred"), 'wt')

        total_num = test_data_loader.get_num_batch()
        test_data_loader.reset()

        for i in range(total_num):
            cur_batch = test_data_loader.get_batch(i)
            print('Instance {}'.format(i))
            line = cur_batch.instances[0][1].tokText.replace(" </s>","")
            ref_outfile.write(line.encode('utf-8') + "\n")
            ref_outfile.flush()
            hyps = beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
            cur_passage = cur_batch.instances[0][0]
            cur_sent = hyps[0].idx_seq_to_string(cur_passage, word_vocab)
            line = cur_sent.replace(" </s>","")
            pred_outfile.write(line.encode('utf-8') + "\n")
            pred_outfile.flush()

        ref_outfile.close()
        pred_outfile.close()