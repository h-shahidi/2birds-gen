import json
import re
import random

import numpy as np

from utils import padding_utils
from utils.sent_utils import Sentence


def read_dataset(inpath, isLower=True):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    
    all_instances = []
    max_question_len = 0
    for instance in dataset:
        instance_id = instance['id']

        passage_tokens = instance['annotation1']['toks_sent']
        if passage_tokens == "": continue
        annotation1 = instance['annotation1']
        annotation1["POSs"] = annotation1["POS_sent"]
        annotation1["NERs"] = annotation1["NER_sent"]
        passage = Sentence(passage_tokens, annotation1, ID_num=instance_id, isLower=False)
        passage.answer_span = instance['IO_sent']

        question_tokens = instance['annotation2']['toks']
        if question_tokens == "": continue
        annotation2 = instance['annotation2']
        question = Sentence(question_tokens, annotation2, ID_num=instance_id, isLower=isLower, end_sym='</s>')
        max_question_len = max(max_question_len, question.get_length()) # text2 is the sequence to be generated

        answer_tokens = instance['annotation3']['toks']
        annotation3 = instance['annotation3']
        answer = Sentence(answer_tokens, annotation3, ID_num=instance_id, isLower=isLower)

        all_instances.append((passage, question, answer))

    return all_instances, max_question_len


def collect_vocabs(all_instances):
    all_words = set()
    all_POSs = set()
    all_NERs = set()
    for (passage, question, answer) in all_instances:
        sentences = [passage, question, answer]
        for sentence in sentences:
            all_words.update(re.split("\\s+", sentence.tokText))
            if sentence.POSs != None and sentence.POSs != []:
                all_POSs.update(re.split("\\s+", sentence.POSs))
            if sentence.NERs != None and sentence.NERs != []:
                all_NERs.update(re.split("\\s+", sentence.NERs))

    return (all_words, all_POSs, all_NERs)


class QGDataLoader(object):
    def __init__(self, all_instances, word_vocab=None, POS_vocab=None, NER_vocab=None, flags=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.flags = flags
        if batch_size == -1: 
            batch_size = flags.batch_size

        processed_instances = []
        for (passage, question, answer) in all_instances:
            if passage.get_length() > flags.max_passage_len: continue # remove very long passages
            if question.get_length() < 3: continue # filter out very short questions
            passage.convert2index(word_vocab, POS_vocab, NER_vocab)
            question.convert2index(word_vocab, POS_vocab, NER_vocab)
            answer.convert2index(word_vocab, POS_vocab, NER_vocab)

            processed_instances.append((passage, question, answer))

        all_instances = processed_instances
        processed_instances = None

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda question: (question[0].get_length(), question[1].get_length()))
        else:
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = SquadBatch(
                cur_instances, flags, word_vocab=word_vocab,
                POS_vocab=POS_vocab, NER_vocab=NER_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]

class SquadBatch(object):
    def __init__(self, instances, flags, word_vocab=None, POS_vocab=None, NER_vocab=None):
        self.flags = flags
        self.instances = instances
        self.batch_size = len(instances)
        self.passage_words = [instances[i][0].tokText.split() for i in range(self.batch_size)]

        # create length
        self.sent1_length = [] # [batch_size]
        self.sent2_length = [] # [batch_size]
        self.sent3_length = [] # [batch_size]
        for (sent1, sent2, sent3) in instances:
            self.sent1_length.append(sent1.get_length())
            self.sent2_length.append(sent2.get_length())
            self.sent3_length.append(sent3.get_length())
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)
        self.sent3_length = np.array(self.sent3_length, dtype=np.int32)

        # create word representation
        start_id = word_vocab.getIndex('<s>')
        if flags.with_word:
            self.sent1_word = [] # [batch_size, sent1_len]
            self.sent2_word = [] # [batch_size, sent2_len]
            self.sent2_input_word = []
            self.sent3_word = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_word.append(sent1.word_idx_seq)
                self.sent2_word.append(sent2.word_idx_seq)
                self.sent2_input_word.append([start_id]+sent2.word_idx_seq[:-1])
                self.sent3_word.append(sent3.word_idx_seq)
            self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)
            self.sent2_word = padding_utils.pad_2d_vals(self.sent2_word, len(self.sent2_word), flags.max_question_len)
            self.sent2_input_word = padding_utils.pad_2d_vals(self.sent2_input_word, len(self.sent2_input_word), flags.max_question_len)
            self.sent3_word = padding_utils.pad_2d_vals_no_size(self.sent3_word)

            self.question_words = self.sent2_word
            self.decoder_inputs = self.sent2_input_word
            self.question_lengths = self.sent2_length

        if flags.with_POS:
            self.sent1_POS = [] # [batch_size, sent1_len]
            self.sent2_POS = [] # [batch_size, sent2_len]
            self.sent3_POS = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_POS.append(sent1.POS_idx_seq)
                self.sent2_POS.append(sent2.POS_idx_seq)
                self.sent3_POS.append(sent3.POS_idx_seq)
            self.sent1_POS = padding_utils.pad_2d_vals_no_size(self.sent1_POS)
            self.sent2_POS = padding_utils.pad_2d_vals_no_size(self.sent2_POS)
            self.sent3_POS = padding_utils.pad_2d_vals_no_size(self.sent3_POS)

        if flags.with_NER:
            self.sent1_NER = [] # [batch_size, sent1_len]
            self.sent2_NER = [] # [batch_size, sent2_len]
            self.sent3_NER = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_NER.append(sent1.NER_idx_seq)
                self.sent2_NER.append(sent2.NER_idx_seq)
                self.sent3_NER.append(sent3.NER_idx_seq)
            self.sent1_NER = padding_utils.pad_2d_vals_no_size(self.sent1_NER)
            self.sent2_NER = padding_utils.pad_2d_vals_no_size(self.sent2_NER)
            self.sent3_NER = padding_utils.pad_2d_vals_no_size(self.sent3_NER)

        if flags.with_answer_span:
            self.answer_span = []
            for (sent1, sent2, sent3) in instances:
                self.answer_span.append(sent1.answer_span)
            self.answer_span = padding_utils.pad_2d_vals_no_size(self.answer_span)

