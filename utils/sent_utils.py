import re

class Sentence(object):
    def __init__(self, tokText, annotation=None, ID_num=None, isLower=False, end_sym=None):
        self.tokText = tokText
        self.annotation = annotation
        if end_sym != None:
            self.tokText += ' ' + end_sym
        if isLower:
            self.tokText = self.tokText.lower()
        self.words = re.split("\\s+", self.tokText)

        self.POSs = []
        self.NERs = []
        if annotation != None:
            self.POSs = annotation['POSs']
            self.NERs = annotation['NERs']

        self.length = len(self.words)
        self.ID_num = ID_num

        self.index_convered = False

    def get_length(self):
        return self.length

    def convert2index(self, word_vocab, POS_vocab, NER_vocab):
        if self.index_convered: return # for each sentence, only conver once

        if word_vocab is not None:
            self.word_idx_seq = word_vocab.to_index_sequence(self.tokText)

        if POS_vocab is not None:
            self.POS_idx_seq = POS_vocab.to_index_sequence(self.POSs)

        if NER_vocab is not None:
            self.NER_idx_seq = NER_vocab.to_index_sequence(self.NERs)	

        self.index_convered = True
