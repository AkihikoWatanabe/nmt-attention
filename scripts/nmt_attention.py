# coding=utf-8

import numpy as np
from chainer import cuda, Variable, optimizers, serializers, utils, gradient_check
from chainer import Link, Chain, ChainList
import chainer.linksa as L
import chainer.functions as F
import argparse
from utils.vocab import Vocab



def parse_args():
    # each default parameter is according to the settings of original paper.
    DEF_EPOCHS = 10
    DEF_EMBED = 600
    DEF_MINIBATCH = 80
    DEF_HIDDEN = 1000
    DEF_VOCAB = 30000

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
            "hidden_size",
            type=int,
            default=DEF_HIDDEN,
            help="# of hidden units"
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
    args = p.parse_args()

    return args

