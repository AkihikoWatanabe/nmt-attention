# coding=utf-8

import argparse
from chainer import cuda, serializers, optimizers
import chainer.functions as F 
from lib.backup import Backup
from lib.vocab import Vocab
from lib.models import AttentionBasedEncoderDecoder as ABED
from lib.generators import word_list, batch, sort
from lib.constants import BEGIN, END
from lib.functions import fill_batch
from lib.XP import XP

HPARAM_NAME = "hyper_params"
TAR_VOCAB_NAME = "tarvocab"
SRC_VOCAB_NAME = "srcvocab"

def forward(src_batch, tar_batch, src_vocab, tar_vocab, encdec, is_train, limit):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    tar_len = len(tar_batch[0]) if tar_batch!=None else 0
    encdec.reset(batch_size)

    # forward encoding
    x = XP.iarray([src_vocab.s2i(BEGIN) for _ in range(batch_size)])
    encdec.fencode(x)
    for l in range(src_len):
        x = XP.iarray([src_vocab.s2i(src_batch[k][l]) for k in range(batch_size)]) 
        encdec.fencode(x)
    x = XP.iarray([src_vocab.s2i(END) for _ in range(batch_size)])
    encdec.fencode(x) 

    # backward encoding
    x = XP.iarray([src_vocab.s2i(END) for _ in range(batch_size)])
    encdec.bencode(x)
    for l in reversed(range(src_len)):
        x = XP.iarray([src_vocab.s2i(src_batch[k][l]) for k in range(batch_size)]) 
        encdec.bencode(x)
    x = XP.iarray([src_vocab.s2i(BEGIN) for _ in range(batch_size)])
    encdec.bencode(x)  

    # initialize states of the decoder
    encdec.init_decode()

    # decoding
    t = XP.iarray([tar_vocab.s2i(BEGIN) for _ in range(batch_size)]) 
    hyp_batch = [[] for _ in range(batch_size)]

    if is_train:
        loss = XP.zeros(())
        for l in range(tar_len):
            y = encdec.decode(t, batch_size)
            t = XP.iarray([tar_vocab.s2i(tar_batch[k][l]) for k in xrange(batch_size)])
            loss += F.softmax_cross_entropy(y, t)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                hyp_batch[k].append(tar_vocab.i2s(output[k]))
        return hyp_batch, loss
    else:
        while len(hyp_batch[0])<limit:
            y = encdec.decode(t, batch_size)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                hyp_batch[k].append(tar_vocab.i2s(output[k]))
            if all(hyp_batch[k][-1]==END for k in xrange(batch_size)):
                break
        return hyp_batch

def train(args):
    source_vocab = Vocab(args.source, args.vocab)
    target_vocab = Vocab(args.target, args.vocab)
    att_encdec = ABED(args.vocab, args.hidden_size, args.maxout_hidden_size, args.embed_size)
    if args.use_gpu:
        att_encdec.to_gpu()
    for epoch in xrange(args.epochs):
        print "--- epoch: %s/%s ---"%(epoch+1, args.epochs)
        source_gen = word_list(args.source)
        target_gen = word_list(args.target)
        batch_gen = batch(sort(source_gen, target_gen, 100*args.minibatch), args.minibatch)
        opt = optimizers.AdaDelta(args.rho, args.eps)
        opt.setup(att_encdec)
        n = 0
        for source_batch, target_batch in batch_gen:
            n += len(source_batch)
            source_batch = fill_batch(source_batch)
            target_batch = fill_batch(target_batch)
            hyp_batch, loss = forward(source_batch, target_batch, source_vocab, target_vocab, att_encdec, True, 0)

            loss.backward()
            opt.weight_decay(0.001)
            opt.update()
        prefix = args.model_path + '%s'%(epoch+1)
        serializers.save_hdf5(prefix+'.attencdec', att_encdec)
    hyp_params = att_encdec.get_hyper_params()
    Backup.dump(hyp_params, args.model_path+HPARAM_NAME)
    source_vocab.save(args.model_path+SRC_VOCAB_NAME)
    target_vocab.save(args.model_path+TAR_VOCAB_NAME)

def test(args):
    source_vocab = Vocab.load(args.model_path+SRC_VOCAB_NAME)
    target_vocab= Vocab.load(args.model_path+TAR_VOCAB_NAME) 
    vocab_size, hidden_size, maxout_hidden_size, embed_size = Backup.load(args.model_path+HPARAM_NAME)

    att_encdec = ABED(vocab_size, hidden_size, maxout_hidden_size, embed_size)
    if args.use_gpu:
        att_encdec.to_gpu()
    serializers.load_hdf5(args.model_path+str(args.epochs)+'.attencdec', att_encdec)

    with open(args.output+str(args.epochs), 'w') as fp:
        source_gen = word_list(args.source)
        target_gen = word_list(args.target)
        batch_gen = batch(sort(source_gen, target_gen, 100*args.minibatch), args.minibatch) 
        for source_batch, target_batch in batch_gen: 
            source_batch = fill_batch(source_batch)
            target_batch = fill_batch(target_batch) 
            hyp_batch = forward(source_batch, None, source_vocab, target_vocab, att_encdec, False, args.limit)
            for i, hyp in enumerate(hyp_batch):
                hyp = hyp[:hyp.index(END)]
                print '--------------------'
                print 'source: %s'%' '.join(source_batch[i])
                fp.write('source: %s\n'%' '.join(source_batch[i]) )
                print 'target: %s'%' '.join(target_batch[i])
                fp.write('target: %s\n'%' '.join(target_batch[i]) ) 
                print 'hyp: %s'%' '.join(hyp)
                fp.write('hyp: %s'%' '.join(hyp))

def parse_args():
    # each default parameter is according to the settings of original paper.
    DEF_EPOCHS = 10
    DEF_EMBED = 620
    DEF_MINIBATCH = 80
    DEF_HIDDEN = 1000
    DEF_MAXOUT_HIDDEN = 500
    DEF_VOCAB = 30000
    DEF_EPS = 1e-06
    DEF_RHO = 0.95
    DEF_OUTPUT = "./hyp"
    DEF_LIMIT = 50

    p = argparse.ArgumentParser(
        description = "A Neural Attention Model for Machine Translation"
            )
    p.add_argument(
            "source",
            type=str,
            help="path_to_source_corpus"
            )
    p.add_argument(
            "target",
            type=str,
            help="path_to_target_corpus"
            )
    p.add_argument(
            "model_path",
            type=str,
            help="path_to_model/ This directory will use save/load model files in training/testing."
            )
    p.add_argument(
            "-output",
            type=str,
            default=DEF_OUTPUT,
            help="path_to_output_file"
            )
    p.add_argument(
            "--train",
            action="store_true",
            help="if set this option, the network will be trained and generate model files."
    )
    p.add_argument(
            "--test",
            action="store_true",
            help="if set this option, decoding on the test data will be conducted using trained models."
            )
    p.add_argument(
            "--use_gpu",
            action="store_true",
            help="using gpu for calculation"
            )
    p.add_argument(
            "-epochs",
            type=int,
            default=DEF_EPOCHS,
            help="# of epochs"
            )
    p.add_argument(
            "-embed_size",
            type=int,
            default=DEF_EMBED,
            help="size of word embedding"
            )
    p.add_argument(
            "-hidden_size",
            type=int,
            default=DEF_HIDDEN,
            help="# of hidden units"
            )
    p.add_argument(
            "-maxout_hidden_size",
            type=int,
            default=DEF_MAXOUT_HIDDEN,
            help="# of maxout hidden units"
            )
    p.add_argument(
            "-minibatch",
            type=int,
            default=DEF_MINIBATCH,
            help="size of miniatch"
            )
    p.add_argument(
            "-vocab",
            type=int,
            default=DEF_VOCAB,
            help="size of vocabulary"
            )
    p.add_argument(
            "-rho",
            type=int,
            default=DEF_RHO,
            help="rho of AdaDelta"
            )
    p.add_argument(
            "-eps",
            type=int,
            default=DEF_EPS,
            help="epsilon of AdaDelta"
            )
    p.add_argument(
            "-limit",
            type=int,
            default=DEF_LIMIT,
            help="maximum number of words of output"
            )
    args = p.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()
    XP.set_library(args.use_gpu)
    if args.train:
        train(args)
    elif args.test:
        test(args)
