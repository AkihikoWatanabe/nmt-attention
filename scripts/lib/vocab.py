# coding=utf-8

from collections import defaultdict

class Vocab():
    def __init__(self, file_path, vocab_size):
        self.__file_path = file_path
        self.__vocab_size = vocab_size

    def save(self, filepath):
        with open(filepath, "w") as fp:
            fp.write('\n'.join(map(str, [self.__file_path, self.__vocab_size, self.__num_lines, self.__num_words])))
            fp.write('\n')
            for s, i in self.__s2i.items():
                fp.write('%s	%s\n'%(s, i))
    
    # TODO: static methodにする
    def load(self, filepath):
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
            return self.__s2i['<unk>'] # 0
        
    def i2s(self, word_id):
        try:
            return self.__i2s[word_id]
        except KeyError:
            return self.__i2s[0] # <unk>

    def make_vocab(self):
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
        # 0:<unk>  1:<s>  2:</s>
        self.__s2i = defaultdict(lambda: len(self.__s2i)+3)
        [self.__s2i[k] for k, v in sorted(word_freq.items(), key=lambda x:-x[1])]
        self.__s2i['<unk>'] = 0
        self.__s2i['<s>'] = 1
        self.__s2i['</s>'] = 2
        self.__i2s = ['']*self.__vocab_size
        self.__i2s[0] = '<unk>'
        self.__i2s[1] = '<s>'
        self.__i2s[2] = '</s>'
        for s, i in self.__s2i.items():
            self.__i2s[i] = s
