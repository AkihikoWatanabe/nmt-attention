# coding=utf-8

from collections import defaultdict
from constants import UNK, BEGIN, END, PAD

class Vocab():
    def __init__(self, file_path, vocab_size, make=True):
        self.__file_path = file_path
        self.__vocab_size = vocab_size

        if make:
            word_freq = defaultdict(lambda: 0)
            self.__num_lines = 0 
            self.__num_words = 0
            with open(self.__file_path) as fp:
                for line in fp:
                    words = line.split()
                    self.__num_lines += 1
                    self.__num_words += len(words)
                    for word in words:
                        word_freq[word] += 1
            # 0:<unk>  1:<s>  2:</s> 3:<pad>
            self.__s2i = defaultdict(lambda: len(self.__s2i)+4)
            [self.__s2i[k] for k, v in sorted(word_freq.items(), key=lambda x:-x[1])[:self.__vocab_size]]
            self.__s2i[UNK]= 0
            self.__s2i[BEGIN] = 1
            self.__s2i[END] = 2
            self.__s2i[PAD] = 3
            self.__i2s = ['']*(self.__vocab_size+4)
            self.__i2s[0] = UNK
            self.__i2s[1] = BEGIN
            self.__i2s[2] = END
            self.__i2s[3] = PAD
            for s, i in self.__s2i.items():
                self.__i2s[i] = s

    def save(self, filepath):
        with open(filepath, "w") as fp:
            fp.write('\n'.join(map(str, [self.__file_path, self.__vocab_size, self.__num_lines, self.__num_words])))
            fp.write('\n')
            for s, i in self.__s2i.items():
                fp.write('%s	%s\n'%(s, i))
    
    @staticmethod
    def load(self, filepath):
        self = Vocab(None, None, make=False)
        with open(filepath) as fp:
            self.__file_path = next(fp).strip()
            self.__vocab_size = int(next(fp))
            self.__num_lines = int(next(fp))
            self.__num_words = int(next(fp))
            self.__s2i = defaultdict(lambda: len(self.__s2i))
            self.__i2s = ['']*self.__vocab_size
            for j in xrange(self.__vocab_size):
                try:
                    s, i = next(fp).strip().split()
                    i = int(i)
                except StopIteration:
                    break
                if s:
                    self.__s2i[s] = i
                    self.__i2s[i] = s
        return self

    def set_num_lines(self, num_lines):
        self.__num_lines = num_lines

    def set_num_words(self, num_words):
        self.__num_words = num_words
    
    def set_s2i(self, s2i):
        self.__s2i = s2i

    def set_i2s(self, i2s):
        self.__i2s = i2s

    def get_number_of_lines(self):
        return self.__num_lines

    def get_number_of_words(self):
        return self.__num_words

    def s2i(self, surface):
        try:
            return self.__s2i[surface]
        except KeyError:
            return self.__s2i[UNK] # 0
        
    def i2s(self, word_id):
        try:
            return self.__i2s[word_id]
        except KeyError:
            return self.__i2s[0] # <unk>
