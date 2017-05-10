# -*- coding: utf-8 -*-
# file: data_loader.py
# author: JinTian
# time: 10/05/2017 6:27 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
this file load pair data into seq2seq model
raw file contains:
sequenceA   sequenceB
....

"""
import torch
from torch.autograd import Variable
import math
import random
import re
import time
import unicodedata
from io import open
from config.global_config import *


class PairDataLoader(object):
    """
    this class load raw file and generate pair data.
    """

    def __init__(self):

        self.SOS_token = 0
        self.EOS_token = 1
        self.eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

        self._prepare_data('eng', 'fra')

    class Lang(object):

        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS

        def add_sentence(self, sentence):
            for word in sentence.split(' '):
                self.add_word(word)

        def add_word(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def filter_pair(self, p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
               len(p[1].split(' ')) < MAX_LENGTH and \
               p[0].startswith(self.eng_prefixes)

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s).lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def read_lang(self, lang1, lang2, reverse=False):
        print("Reading lines...")
        lines = open('./datasets/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
            read().strip().split('\n')
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = self.Lang(lang2)
            output_lang = self.Lang(lang1)
        else:
            input_lang = self.Lang(lang1)
            output_lang = self.Lang(lang2)

        return input_lang, output_lang, pairs

    @staticmethod
    def indexes_from_sentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variable_from_sentence(self, lang, sentence):
        indexes = self.indexes_from_sentence(lang, sentence)
        indexes.append(self.EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def _prepare_data(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.read_lang(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        self.pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(self.pairs))
        print("Counting words...")
        for pair in self.pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        self.input_lang = input_lang
        self.output_lang = output_lang
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

    def get_pair_variable(self):
        input_variable = self.variable_from_sentence(self.input_lang, random.choice(self.pairs)[0])
        target_variable = self.variable_from_sentence(self.output_lang, random.choice(self.pairs)[1])
        return input_variable, target_variable

