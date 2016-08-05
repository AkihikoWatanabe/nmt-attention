# coding=utf-8

import numpy as np
from chainer import cuda, Variable, optimizers, serializers, utils, gradient_check
from chainer import Link, Chain, ChainList
import chainer.linksa as L
import chainer.functions as F
from vocab import Vocab

class LSTMEncoder(Chain):
    def __init__(self, vocab_size, hidden_size, embed_size):
        super(Encoder, self).__init__(
            eh = L.Linear(embed_size, 4*hidden_size),
            hh = L.Linear(hidden_size, 4*hidden_size),
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

    def __call__(self, e, c, h):
        c, h = F.lstm(c, self.eh(e)+self.hh(h))
        return c, h

class BiDirectionalLSTMEncoder(Chain):
    """ This class encodes inputs as embedding bidirectionally.
        See the paper Appendix A.2.1 for more details.
        This implementation uses LSTM as gated hidden unit instead of GRU.
    """
    def __init__(self, vocab_size, hidden_size, embed_size, vocab):
        super(BiDirectionalLSTMEncoder, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size),
            fenc = LSTMEncoder(vocab_size, hidden_size, embed_size),
            benc = LSTMEncoder(vocab_size, hidden_size, embed_size),
        )
        self.vocab = vocab
        
    def __call__(self, x, fc, fh, bc, bh):
        """ @param:
                x: input words
                fc: memory cell for forward RNN
                fh: previous hidden state for forward RNN
                bc: memory cell for backward RNN
                bh: previous hidden state for backward RNN
            @return:
                h: hidden states for Encoder
        """
        fh = []
        bh = []
        src_len = len(x)
        # forward

        # backward
        
class Alignment(Chain):
    """ This class calculate the alignment score:
        eij = a(s_{i-1}, h_{j}) = v_{a}^{T} * tanh(W_{a}s_{i-1} + U_{a}h_{j})
        See the paper Appendix A.1.2 for more details.
    """
    def __init__(self, hidden_size):
        super(Alignment, self).__init__(
            ws = L.Linear(hidden_size, hidden_size),
            uh = L.Linear(2*hidden_size, hidden_size),
            vh = L.Linear(hidden_size, 1),
        )
    def __call__(self, s, h):
        """ @param:
                s_{i-1}: (i-1)-th state of the Decoder
                h_{j}: j-th state of the Encoder
            @return:
                e_{ij}: alignment score between target position at i and input position at j before softmax
        """
        t = F.tanh(self.ws(s)+self.uh(h))
        e = self.vh(t)
        return e

class Decoder(Chain):

    def __init__(self, vocab_size, hidden_size, embed_size):
        super(Decoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4*hidden_size),
            cc = L.Linear(2*hidden_size, 4*hidden_size),

        )

