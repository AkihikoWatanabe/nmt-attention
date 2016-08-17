# coding=utf-8

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def fill_batch_zero(batch, size):
    return [x+[0.0]*(size-len(x)) for x in batch]

def fill_batch_end(batch, token='</s>'):
    max_len = max([len(x) for x in batch])
    return [x+[token]*(max_len-len(x)+1) for x in batch]

def fill_batch_startend(batch, start_token='<s>', end_token='</s>'):
    max_len = max([len(x) for x in batch])
    return [[start_token]+x+[end_token]*(max_len-len(x)+1) for x in batch]

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
