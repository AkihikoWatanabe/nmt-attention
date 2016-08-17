# coding=utf-8
import math

def word_list(filepath):
    with open(filepath) as fp:
        for line in fp:
            yield line.split()

def batch(generator, batch_size):
    batch = []
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch)==batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sort(generator1, generator2, pooling, sort_target=1):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[sort_target])):
            yield x
