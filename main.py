# This code is based on https://github.com/tyliupku/wiki2bio.
# We would like to thank the authors for sharing their code base.

import sys
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import * 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 1480,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 5000,'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0005,'learning rate')

tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("loss", 'ce', 'cross entropy (ce) or reinforcement learning (rl)')
tf.app.flags.DEFINE_string("load",'0','load directory') 
tf.app.flags.DEFINE_string("dir",'processed_data','data set directory')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

# test phase
if FLAGS.load != "0":
    save_dir = 'results/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    prefix = str(int(time.time() * 1000))
    save_dir = 'results/res/' + prefix + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + prefix + '/'
    os.mkdir(save_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'


def train(sess, dataloader, model):
    write_log("#######################################################")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    vocab = Vocab()
    k = 0
    loss, start_time = 0.0, time.time()
    for _ in range(FLAGS.epoch):
        for batch in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
            if FLAGS.loss == 'ce':
                loss += model.ce_train(batch, sess)
            else:
                loss += model.rl_train(batch, sess, vocab)
            k += 1
            progress_bar(k%FLAGS.report, FLAGS.report)
            if (k % FLAGS.report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                sess.run(model.ema_to_vars_op)
                if k // FLAGS.report >= 1: 
                    ksave_dir = save_model(model, save_dir, k // FLAGS.report)
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
                sess.run(model.restore_backup_vars_op)


def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        texts_path = "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        texts_path = "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set
    
    # for copying words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    vocab = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    k = 0
    for batch in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
        predictions, atts = model.generate(batch, sess)
        atts = np.squeeze(atts)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(vocab.id2word(tid))
                        mask_sum.append(vocab.id2word(tid))
                    unk_sum.append(vocab.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))

    result = copy_result + nocopy_result 
    print result

    return result

def visualize_attn(sess, dataloader, model):
    texts_path = "processed_data/test/test.box.val"
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    vocab = Vocab()
    evalset = dataloader.test_set
    idx = 56
    i = 0
    for batch in dataloader.batch_iter(evalset, 1, False):
        if i >= idx:
            break
        i += 1
    predictions, atts = model.generate(batch, sess)
    atts = np.squeeze(atts)
    predictions = np.squeeze(predictions)

    summary = list(predictions)
    if 2 in summary:
        summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
    generated_summary= []
    for tk, tid in enumerate(summary):
        if tid == 3:
            sub = texts[idx][np.argmax(atts[tk, :len(texts[idx])])]
            generated_summary.append(sub)
        else:
            generated_summary.append(vocab.id2word(tid))
    print(texts[idx])
    print(generated_summary)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(atts, cmap=plt.cm.Purples)
    fig.colorbar(cax)
    box_vals = []
    texts[idx][0] = 'frederic'
    texts[idx][-2] = 'frederic'
    generated_summary[0] = 'frederic'
    # Set up axes
    ax.set_xticklabels([''] + texts[idx], rotation=90, fontdict={'fontsize':16})
    ax.set_yticklabels([''] + generated_summary + ['<eos>'], fontdict={'fontsize':16})
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def write_log(s):
    print s
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir)
    return nnew_dir


def main(_):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    is_training = FLAGS.mode == 'train'
    with tf.Session(config=config) as sess:
        copy_file(save_file_dir)
        dataloader = DataLoader(FLAGS.dir)
        model = SeqUnit(FLAGS, scope_name="seq2seq", is_training=is_training)
        sess.run(tf.global_variables_initializer())
        if FLAGS.load != '0':
            tvars = tf.trainable_variables()
            model.load(save_dir)
            sess.run(tf.group(*(tf.assign(model.ema.average(var), var) for var in tvars)))
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)
        elif FLAGS.mode == "visualize":
            visualize_attn(sess, dataloader, model)
        else:
            evaluate(sess, dataloader, model, save_dir, mode='test')


if __name__=='__main__':
    tf.app.run(main=main)
