# coding=utf-8

import argparse
from lib.backup import Backup
from lib.vocab import Vocab
from lib.models import AttentionBasedEncoderDecoder as ABED
from lib.generators import word_list, batch, sort
from chainer import serializers

HPARAM_NAME = "hyper_params"
TRG_VOCAB_NAME = "trgvocab"
SRC_VOCAB_NAME = "srcvocab"

def forward():

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
            source_batch = fill_batch_end(source_batch)
            target_batch = fill_batch_end(target_batch)
            hyp_batch, loss = forward()
            
            loss.backward()
            opt.weight_decay(0.001)
            opt.update()
        prefix = args.model_path + '%s'%(epoch+1)
        serializers.save_hdf5(prefix+'.attencdec', att_encdec)
    hyp_params = att_encdec.get_hyper_params()
    Backup.dump(args.model_path+HPARAM_NAME)
    source_vocab.save(args.model_path+SRC_VOCAB_NAME)
    target_vocab.save(args.model_path+TRG_VOCAB_NAME)

def test(args):
    

def parse_args():
    # each default parameter is according to the settings of original paper.
    DEF_EPOCHS = 10
    DEF_EMBED = 620
    DEF_MINIBATCH = 80
    DEF_HIDDEN = 1000
    DEF_MAXOUT_HIDDEN = 500
    DEF_VOCAB = 30000
    DEF_EPS = 1e-06
    DFE_RHO = 0.95

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
    args = p.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()
    XP.set_library(args.use_gpu)
    if args.train:
        train(args)
    elif args.test:
        test(args)
