# coding=utf-8

from constants import BEGIN, END, PAD

def fill_batch_zero(batch, size):
    return [x+[0.0]*(size-len(x)) for x in batch]

def fill_batch(batch, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [x+[pad]*(max_len-len(x)) for x in batch] 

def fill_batch_end(batch, end_token=END, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [x+[end_token]+[pad]*(max_len-len(x)) for x in batch]

def fill_batch_begin_end(batch, start_token=BEGIN, end_token=END, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [[BEGIN]+x+[end_token]+[pad]*(max_len-len(x)) for x in batch] 

def fill_batch_startend(batch, start_token=BEGIN, end_token=END, pad=PAD):
    max_len = max([len(x) for x in batch])
    return [[start_token]+x+[end_token]+[pad]*(max_len-len(x)) for x in batch]
