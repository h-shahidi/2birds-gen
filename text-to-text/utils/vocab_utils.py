from __future__ import print_function
import re

import numpy as np
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer

class Vocab(object):
    def __init__(self, embedding_path=None, vocab=None, dim=100, word2id=None, word_vecs=None, unk_mapping_path=None):
        self.t = TweetTokenizer()
        self.unk_label = '<unk>'
        if embedding_path != None:
            self.fromText(embedding_path, vocab=vocab, pre_word_vecs=word_vecs)
        else: # build a vocabulary with a word set
            self.fromVocabualry(vocab, dim=dim)

        self.__unk_mapping = None
        if unk_mapping_path is not None:
            self.__unk_mapping = {}
            in_file = open(unk_mapping_path, 'rt')
            for line in in_file:
                items = re.split('\t', line)
                self.__unk_mapping[items[0]] = items[1]
            in_file.close()


    def fromVocabualry(self, voc, dim=100):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        voc.discard("")
        self.vocab_size = len(voc)
        self.word_dim = dim
        for word in voc:
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word

        shape = (self.vocab_size+1, self.word_dim)
        scale = 0.05
        self.word_vecs = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)

    def fromText(self, vec_path, vocab=None, pre_word_vecs=None):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}

        vec_file = open(vec_path, 'rt')
        word_vecs = {}
        for i, line in enumerate(vec_file):
            line = line.decode('utf-8').strip()
            parts = line.split('\t')
            cur_index = int(parts[0])
            word = parts[1]
            vector = np.array(map(float,re.split('\\s+', parts[2])), dtype='float32')
            assert word not in self.word2id, word
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
            self.word_dim = vector.size
        vec_file.close()

        self.vocab_size = len(self.word2id)

        if pre_word_vecs is not None:
            self.word_vecs = pre_word_vecs
        else:
            self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
            for cur_index in xrange(self.vocab_size):
                self.word_vecs[cur_index] = word_vecs[cur_index]


    def setWordvec(self,word_vecs):
        self.word_vecs = word_vecs

    def hasWord(self, word):
        return self.word2id.has_key(word)

    def size(self):
        return len(self.word2id)

    def getIndex(self, word):
        if(self.word2id.has_key(word)):
            return self.word2id.get(word)
        else:
            return self.vocab_size

    def getWord(self, idx):
        return 'UNK' if idx == self.vocab_size else self.id2word.get(idx)

    def getVector(self, word):
        if(self.word2id.has_key(word)):
            idx = self.word2id.get(word)
            return self.word_vecs[idx]
        return None

    def getLexical(self, sout):
        end_id = self.getIndex('</s>')
        try:
            k = sout.index(end_id)
            sout = sout[:k]
        except ValueError:
            pass
        slex = ' '.join([self.getWord(x) for x in sout])
        return sout, slex


    def to_index_sequence(self, sentence):
        sentence = sentence.strip()
        seq = []
        for i, word in enumerate(re.split('\\s+', sentence)):
            idx = self.getIndex(word)
            if idx == None and self.__unk_mapping is not None and self.__unk_mapping.has_key(word):
                simWord = self.__unk_mapping[word]
                idx = self.getIndex(simWord)
            if idx == None: idx = self.vocab_size
            seq.append(idx)
        return seq

    def dump_to_txt(self, outpath):
        outfile = open(outpath, 'wt')
        for word in self.word2id.keys():
            cur_id = self.word2id[word]
            cur_vector = self.getVector(word)
            word= word.encode('utf-8')
            outline = "{}\t{}\t{}".format(cur_id, word, vec2string(cur_vector))
            outfile.write(outline + "\n")
        outfile.close()

def vec2string(val):
    result = ""
    for v in val:
        result += " {}".format(v)
    return result.strip()

