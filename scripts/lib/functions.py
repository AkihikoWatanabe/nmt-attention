# coding=utf-8

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from constants import BEGIN, END, PAD

def fill_batch_zero(batch, size):
    return [x+[0.0]*(size-len(x)) for x in batch]

def fill_batch(batch, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [x+[pad]*(max_len-len(x)) for x in batch] 

def fill_batch_end(batch, end_token=END, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [x+[end_token]+[pad]*(max_len-len(x)) for x in batch]

def fill_batch_startend(batch, start_token=BEGIN, end_token=END, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [[start_token]+x+[end_token]+[pad]*(max_len-len(x)) for x in batch]

def bleu(ref, hyp):
    return sentence_bleu(ref, hyp)

def bleu_corpus(ref, hyp):
    return corpus_bleu(ref, hyp)

def normalize(x, lib):
    col = x.shape(1)
    mu = lib.mean(x, axis=0)
    sigma = lib.std(x, axis=0)
    
    for i in xrange(col):
        x[:,i] = (x[:,i]-mu[i])/sigma[i]
    return x
