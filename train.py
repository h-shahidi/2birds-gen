from __future__ import print_function

import random
import os
import argparse
import sys
import time
import codecs

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
cc = SmoothingFunction()

from utils.vocab_utils import Vocab
from utils import config_utils
from modules.data import read_data_split_1, read_data_split_2, collect_vocabs, QGDataLoader
from modules.model import ModelGraph

FLAGS = None
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


def document_bleu(vocab, gen, ref, suffix=''):
    genlex = [vocab.getLexical(x)[1] for x in gen]
    reflex = [[vocab.getLexical(x)[1]] for x in ref]
    genlst = [x.split() for x in genlex]
    reflst = [[x[0].split()] for x in reflex]
    f = codecs.open('gen.txt'+suffix,'w','utf-8')
    for line in genlex:
        print(line, end='\n', file=f)
    f.close()
    f = codecs.open('ref.txt'+suffix,'w','utf-8')
    for line in reflex:
        print(line[0], end='\n', file=f)
    f.close()
    return corpus_bleu(reflst, genlst, smoothing_function=cc.method3)


def evaluate(sess, valid_graph, dev_data_loader, flags=None, suffix=''):
    dev_data_loader.reset()
    gen = []
    ref = []
    dev_loss = 0.0
    dev_right = 0.0
    dev_total = 0.0
    for batch_index in xrange(dev_data_loader.get_num_batch()): # for each batch
        cur_batch = dev_data_loader.get_batch(batch_index)
        if valid_graph.mode == 'evaluate':
            accu_value, loss_value = valid_graph.ce_train(sess, cur_batch, only_eval=True)
            dev_loss += loss_value
            dev_right += accu_value
            dev_total += np.sum(cur_batch.question_lengths)
        elif valid_graph.mode == 'evaluate_bleu':
            gen.extend(valid_graph.run_greedy(sess, cur_batch).tolist())
            ref.extend(cur_batch.question_words.tolist())
        else:
            raise ValueError

    if valid_graph.mode == 'evaluate':
        return {'dev_loss':dev_loss, 'dev_accu':1.0*dev_right/dev_total, 'dev_right':dev_right, 'dev_total':dev_total}
    else:
        return {'dev_bleu':document_bleu(valid_graph.word_vocab, gen, ref, suffix)}


def main(_):
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/"
    log_file_path = path_prefix + "log.txt"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    config_utils.save_config(FLAGS, path_prefix + "config.json")

    print('Loading train set.')
    if FLAGS.data_split == 1:
        train_set, train_question_len = read_data_split_1(FLAGS.s1_train_path, isLower=FLAGS.isLower)
    else:
        train_set, train_question_len = read_data_split_2(FLAGS.s2_train_path, isLower=FLAGS.isLower)
    print('Number of training samples: {}'.format(len(train_set)))

    print('Loading test set.')
    if FLAGS.data_split == 1:
        dev_set, dev_question_len = read_data_split_1(FLAGS.s1_dev_path, isLower=FLAGS.isLower)
    else:
        dev_set, dev_question_len = read_data_split_2(FLAGS.s2_dev_path, isLower=FLAGS.isLower)
    print('Number of test samples: {}'.format(len(dev_set)))

    max_actual_len = max(train_question_len, dev_question_len)
    print('Max answer length: {}, truncated to {}'.format(max_actual_len, FLAGS.max_question_len))

    word_vocab = None
    POS_vocab = None
    NER_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + "best.model"
    if os.path.exists(best_path + ".index"):
        has_pretrained_model = True
        print('There is an existing pretrained model. Loading vocabs:')
        if FLAGS.with_word:
            word_vocab = Vocab(embedding_path=FLAGS.word_vec_path)
            print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        if FLAGS.with_POS:
            POS_vocab = Vocab(embedding_path=os.path.join(path_prefix, "POS_vocab"))
            print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
        if FLAGS.with_NER:
            NER_vocab = Vocab(embedding_path=os.path.join(path_prefix, "NER_vocab"))
            print('NER_vocab: {}'.format(NER_vocab.word_vecs.shape))

    else:
        print('Collecting vocabs.')
        (allWords, allPOSs, allNERs) = collect_vocabs(train_set)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of allPOSs: {}'.format(len(allPOSs)))
        print('Number of allNERs: {}'.format(len(allNERs)))

        if FLAGS.with_word:
            word_vocab = Vocab(embedding_path=FLAGS.word_vec_path)
        if FLAGS.with_POS:
            POS_vocab = Vocab(vocab=allPOSs, dim=FLAGS.POS_dim)
            POS_vocab.dump_to_txt(os.path.join(path_prefix, "POS_vocab"))
        if FLAGS.with_NER:
            NER_vocab = Vocab(vocab=allNERs, dim=FLAGS.NER_dim)
            NER_vocab.dump_to_txt(os.path.join(path_prefix, "NER_vocab"))
        
    print('word vocab size {}'.format(word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build data loaders ... ')
    train_data_loader = QGDataLoader(train_set, word_vocab, POS_vocab, NER_vocab, flags=FLAGS,
                                   isShuffle=True, isLoop=True, isSort=True)

    dev_data_loader = QGDataLoader(dev_set, word_vocab, POS_vocab, NER_vocab, flags=FLAGS,
                                 isShuffle=False, isLoop=False, isSort=True)
    print('Number of instances in train data loader: {}'.format(train_data_loader.get_num_instance()))
    print('Number of instances in dev data loader: {}'.format(dev_data_loader.get_num_instance()))
    sys.stdout.flush()

    # initialize the best bleu and accu scores for current training session
    best_accu = FLAGS.best_accu if 'best_accu' in FLAGS.__dict__ else 0.0
    best_bleu = FLAGS.best_bleu if 'best_bleu' in FLAGS.__dict__ else 0.0
    if best_accu > 0.0:
        print('With initial dev accuracy {}'.format(best_accu))
    if best_bleu > 0.0:
        print('With initial dev BLEU score {}'.format(best_bleu))

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(word_vocab=word_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, flags=FLAGS, mode=FLAGS.mode)

        assert FLAGS.mode in ('ce_train', 'rl_train', 'rl_ce_train')
        valid_mode = 'evaluate' if FLAGS.mode == 'ce_train' else 'evaluate_bleu'

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, flags=FLAGS, mode=valid_mode)

        initializer = tf.global_variables_initializer()

        _vars = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            _vars[var.name.split(":")[0]] = var
        saver = tf.train.Saver(_vars)

        config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=config)
        sess.run(initializer)

        if has_pretrained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")
            
            if FLAGS.mode in ('rl_train','rl_ce_train') and abs(best_bleu) < 0.00001:
                print("Getting BLEU score for the model")
                best_bleu = evaluate(sess, valid_graph, dev_data_loader, flags=FLAGS)['dev_bleu']
                FLAGS.best_bleu = best_bleu
                config_utils.save_config(FLAGS, path_prefix + "config.json")
                print('BLEU = %.4f' % best_bleu)
                log_file.write('BLEU = %.4f\n' % best_bleu)
            if FLAGS.mode == 'ce_train' and abs(best_accu) < 0.00001:
                print("Getting ACCU score for the model")
                best_accu = evaluate(sess, valid_graph, dev_data_loader, flags=FLAGS)['dev_accu']
                FLAGS.best_accu = best_accu
                config_utils.save_config(FLAGS, path_prefix + "config.json")
                print('ACCU = %.4f' % best_accu)
                log_file.write('ACCU = %.4f\n' % best_accu)

        print('Start the training loop.')
        train_size = train_data_loader.get_num_batch()
        max_steps = train_size * FLAGS.n_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            cur_batch = train_data_loader.nextBatch()
            if FLAGS.mode == 'rl_train':
                loss_value = train_graph.rl_train(sess, cur_batch, with_ce=False)
            elif FLAGS.mode == 'rl_ce_train':
                loss_value = train_graph.rl_train(sess, cur_batch, with_ce=True)
            elif FLAGS.mode == 'ce_train':
                loss_value = train_graph.ce_train(sess, cur_batch)
            total_loss += loss_value

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % train_data_loader.get_num_batch() == 0 or (step + 1) == max_steps:
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                log_file.flush()
                sys.stdout.flush()
                total_loss = 0.0

                # Evaluate against the validation set.
                start_time = time.time()
                sess.run(train_graph.ema_to_vars_op)
                res_dict = evaluate(sess, valid_graph, dev_data_loader, flags=FLAGS, suffix=str(step))
                if valid_graph.mode == 'evaluate':
                    dev_loss = res_dict['dev_loss']
                    dev_accu = res_dict['dev_accu']
                    dev_right = int(res_dict['dev_right'])
                    dev_total = int(res_dict['dev_total'])
                    print('Dev loss = %.4f' % dev_loss)
                    log_file.write('Dev loss = %.4f\n' % dev_loss)
                    print('Dev accu = %.4f %d/%d' % (dev_accu, dev_right, dev_total))
                    log_file.write('Dev accu = %.4f %d/%d\n' % (dev_accu, dev_right, dev_total))
                    log_file.flush()
                    if best_accu < dev_accu:
                        print('Saving weights, ACCU {} (prev_best) < {} (cur)'.format(best_accu, dev_accu))
                        saver.save(sess, best_path)
                        best_accu = dev_accu
                        FLAGS.best_accu = dev_accu
                        config_utils.save_config(FLAGS, path_prefix + "config.json")
                else:
                    dev_bleu = res_dict['dev_bleu']
                    print('Dev bleu = %.4f' % dev_bleu)
                    log_file.write('Dev bleu = %.4f\n' % dev_bleu)
                    log_file.flush()
                    if best_bleu < dev_bleu:
                        print('Saving weights, BLEU {} (prev_best) < {} (cur)'.format(best_bleu, dev_bleu))
                        saver.save(sess, best_path)
                        best_bleu = dev_bleu
                        FLAGS.best_bleu = dev_bleu
                        config_utils.save_config(FLAGS, path_prefix + "config.json")
                sess.run(train_graph.restore_backup_vars_op)
                duration = time.time() - start_time
                print('Duration %.3f sec' % (duration))
                sys.stdout.flush()

                log_file.write('Duration %.3f sec\n' % (duration))
                log_file.flush()

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    parser.add_argument('--data_split', type=int, default=1)

    args = parser.parse_args()

    if args.config_path is not None:
        print('Loading the configuration from ' + args.config_path)
        FLAGS = config_utils.load_config(args)

    tf.app.run(main=main)
