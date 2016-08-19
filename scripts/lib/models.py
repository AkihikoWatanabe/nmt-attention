# coding=utf-8

from chainer import Chain
import chainer.links as L
import chainer.functions as F
from lib.XP import XP
from constants import PAD_IDX

class LSTMEncoder(Chain):
    """ This class encodes the input characters as fixed-length vector
    """
    def __init__(self, vocab_size, hidden_size, embed_size):
        super(LSTMEncoder, self).__init__(
            eh = L.Linear(embed_size, 4*hidden_size),
            hh = L.Linear(hidden_size, 4*hidden_size),
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

    def __call__(self, e, c, h):
        """ @param:
                e: word embedding of inputs
                c: memory cell of LSTM
                h: hidden state of LSTM
            @return:
                c: updated memory cell of LSTM
                h: updated hidden state of LSTM
        """
        c, h = F.lstm(c, self.eh(e)+self.hh(h))
        return c, h

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
        self.hidden_size = hidden_size

    def __call__(self, s, h):
        """ @param:
                s: s_{i-1}, (i-1)-th state of the Decoder
                h: h_{j}, j-th state of the Encoder
            @return:
                e: e_{ij}, alignment score between target position at i and input position at j before softmax
        """
        t = F.tanh(self.ws(s)+self.uh(h))
        e = self.vh(t)
        return e

class Decoder(Chain):
    """ This class decodes outputs of Encoder.
        See the paper Appendix A.2.2 for more details.
        This implementation uses LSTM as gated hidden unit instead of GRU.
    """                                                        
    def __init__(self, vocab_size, hidden_size, maxout_hidden_size, embed_size, pool_size=2):
        super(Decoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=PAD_IDX),
            eh = L.Linear(embed_size, 4*hidden_size),
            ch = L.Linear(2*hidden_size, 4*hidden_size),
            hh = L.Linear(hidden_size, 4*hidden_size),
            # single maxout hidden layer
            sm = L.Linear(hidden_size, 2*maxout_hidden_size),
            em = L.Linear(embed_size, pool_size**maxout_hidden_size),
            cm = L.Linear(2*hidden_size, pool_size*maxout_hidden_size),
            my = L.Linear(maxout_hidden_size, vocab_size),
        )
        self.POOL_SIZE = 2

    def __call__(self, y, cv, c, h):
        """ @param:
                y: y_{i-1}, last generated word
                cv: context vector c_{i}
                c: LSTM memory cell
                h: LSTM hidden state
            @return:
                y: the weight of y_{i}
                c: Updated LSTM memory cell
                h: Updated LSTM hidden state
        """ 
        e = self.ye(y)
        t = F.maxout(self.sm(h)+self.em(y)+self.cm(cv), self.POOL_SIZE)
        y = self.my(t)
        c, h = F.lstm(c, self.eh(e)+self.hh(h)+self.ch(cv))

        return y, c, h 

class AttentionBasedEncoderDecoder(Chain):
    """ This class controlls whole models on Attentional MT (Encoder, Decoder, Attention).
    """
    def __init__(self, vocab_size, hidden_size, maxout_hidden_size, embed_size):
        super(AttentionBasedEncoderDecoder, self).__init__(
            # Bi-directional LSTM Encoder
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=PAD_IDX),
            hs = L.Linear(hidden_size, hidden_size),
            fenc = LSTMEncoder(vocab_size, hidden_size, embed_size),
            benc = LSTMEncoder(vocab_size, hidden_size, embed_size),
            # Alignment model
            align = Alignment(hidden_size),
            # Decoder
            dec = Decoder(vocab_size, hidden_size, maxout_hidden_size, embed_size)
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.maxout_hidden_size = maxout_hidden_size
        self.embed_size = embed_size
        self.fstates = []
        self.bstates = []

    def get_hyper_params(self):
        """ @return:
                [list of hyperparamters]: contains vocabulary size, # of hidden unit, # of maxout hidden unit and size of embedding
        """
        return [self.vocab_size, self.hidden_size, \
                self.maxout_hidden_size, self.embed_size]

    def reset(self, batch_size):
        """ Initialize gradients and states.
            This method should be called before encoding.
            @param:
                batch_size: the batch size of the data
        """
        self.zerograds()
        self.fc = XP.zeros((batch_size, self.hidden_size))
        self.fh = XP.zeros((batch_size, self.hidden_size)) 
        self.bc = XP.zeros((batch_size, self.hidden_size))
        self.bh = XP.zeros((batch_size, self.hidden_size))  
        self.dc = XP.zeros((batch_size, self.hidden_size))
        self.dh = XP.zeros((batch_size, self.hidden_size))   
        self.fstates = []
        self.bstates = []
        self.sentence_length = 0
        
    def fencode(self, x):
        """ Conduct forward encoding.
            @param:
                x: word which consists of the input sentences.
        """
        e = self.xe(x)
        self.fc, self.fh = self.fenc(e, self.fc, self.fh)
        self.sentence_length += 1
        self.fstates.append(self.fh)

    def bencode(self, x):
        """ Conduct backward encoding.
            @param:
                x: word which consists of the input sentences.
        """
        e = self.xe(x)
        self.bc, self.bh = self.benc(e, self.bc, self.bh)
        self.sentence_length += 1
        self.bstates.insert(0, self.bh)
    
    def __attention(self, s, batch_size):
        """ Conduct local attention algorithm and return the context vector.
            @param:
                s: state of decoder
                batch_size: batch size of data
            @return:
                cv: context vector
        """
        e_list = []
        for f, b in zip(self.fstates, self.bstates):
            h = F.concat(f, b)
            e_list.append(self.align(s, h))
        cv = XP.zeros((batch_size, 2*self.hidden_size))
        sum_e = sum(e_list)
        for f, b, e in zip(self.fstates, self.bstates, e_list):
            alpha = e / sum_e
            cv += alpha * F.concat(f, b)
        return cv

    def init_decode(self):
        """ Initialize state of the decoder.
            This method should be called before decoding.
        """
        assert len(self.fstates)==len(self.bstates), ["# of the length on the forward encoder and backward encoder is not same."]
        # see first paragraph on A.2.2 Decoder
        self.dh = F.tanh(self.hs(self.bh))

    def decode(self, y, batch_size):
        """ Conduct decoding.
            @param:
                y: y_{i-1}, the output of the decoder at position i-1.
                batch_size: batch size of the data
            @return:
                y: y_{i}, the output of the decoder at position i.
        """
        cv = self.__attention(self.dh, batch_size)
        y, self.dc, self.dh = self.dec(y, cv, self.dc, self.dh)

        return y
