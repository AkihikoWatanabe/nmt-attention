# coding=utf-8

import argparse
from chainer import cuda, serializers, optimizer, optimizers
import chainer.functions as F 
#from nltk.translate.bleu_score import sentence_bleu, corpus_bleu 
import os
from lib.backup import Backup
from lib.vocab import Vocab
from lib.models import AttentionBasedEncoderDecoder as ABED
from lib.generators import word_list, batch, sort
from lib.constants import BEGIN, END, PAD, DECAY_COEFF, PLOT_DIR, CLIP_THR
from lib.functions import fill_batch_end
from lib.XP import XP
import math

os.environ['PATH'] += ':/usr/local/cuda/bin'

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

    # backward encoding
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
            t = XP.iarray(output)
            for k in range(batch_size):
                hyp_batch[k].append(tar_vocab.i2s(output[k]))
            if all(END in hyp_batch[k] for k in xrange(batch_size)):
                break
        return hyp_batch

# TODO: implements batch-level beam search
def forward_beam(src_batch, tar_batch, src_vocab, tar_vocab, encdec, is_train, limit, beam):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    tar_len = len(tar_batch[0]) if tar_batch!=None else 0
    hyp_batch = []

    for k in range(batch_size):
        encdec.reset(1)
        # forward encoding
        x = XP.iarray([src_vocab.s2i(BEGIN)])
        encdec.fencode(x)
        for l in range(src_len):
            x = XP.iarray([src_vocab.s2i(src_batch[k][l])]) 
            encdec.fencode(x)
        # backward encoding
        for l in reversed(range(src_len)):
            x = XP.iarray([src_vocab.s2i(src_batch[k][l])]) 
            encdec.bencode(x)
        x = XP.iarray([src_vocab.s2i(BEGIN)])
        encdec.bencode(x)
        # initialize states of the decoder
        encdec.init_decode()
        # decoding
        t = XP.iarray([tar_vocab.s2i(BEGIN)]) 

        # beam search
        # forward
        best_score = [{} for _ in range(limit+1)]
        best_edge = [{} for _ in range(limit+1)]
        states = [[] for _ in range(limit+1)]
        active_words = []
        BEGIN_IDX = tar_vocab.s2i(BEGIN)
        END_IDX = tar_vocab.s2i(END)
        best_score[0][BEGIN_IDX] = 0
        best_edge[0][BEGIN_IDX] = None
        states[0].append(encdec.get_decoding_state())
        active_words.append([BEGIN_IDX])
        t = XP.iarray([BEGIN_IDX])
        complete_hyp = []
        for l in range(limit):
            my_best = {}
            for prev in active_words[l]:
                for s in range(len(states[l])):
                    encdec.set_decoding_state(states[l][s])
                    y = encdec.decode(XP.iarray([prev]), 1)
                    states[l+1].append(encdec.get_decoding_state())
                    p = F.softmax(y)
                    for next in range(encdec.vocab_size):
                        if best_score[l].has_key(prev):
                            try:
                                score = best_score[l][prev] - math.log(p.data[0][next])
                            except ValueError:
                                score = float('inf')
                            try:
                                if best_score[l+1][next]>score:
                                    best_score[l+1][next] = score
                                    best_edge[l+1][next] = (l, prev)
                                    my_best[next] = score
                            except KeyError:
                                best_score[l+1][next] = score
                                best_edge[l+1][next] = (l, prev)
                                my_best[next] = score
            active_words.append(map(lambda x:x[0], sorted(my_best.items(), key=lambda x:x[1]))[:beam])
            
            if END_IDX in active_words[-1]:
                complete_hyp.append((l+1, END_IDX))
                active_words[-1].pop(active_words.index(END_IDX))
            if len(complete_hyp)==beam:
                break
        # backward
        hyp_scores = []
        for hyp in complete_hyp:
            hyp_scores.append((hyp[0], best_score[hyp[0], END_IDX]))
        l, _ = sorted(hyp_scores, key=lambda x:x[1])[0]
        e = best_edge[l][END_IDX]
        hyp = []
        while e[0]!=0:
            hyp.insert(0, tar_vocab.i2s(e[1]))
            e = best_edge[e[0]][e[1]]
        hyp_batch.append(hyp)

    return hyp_batch


def train(args):
    source_vocab = Vocab(args.source, args.vocab)
    target_vocab = Vocab(args.target, args.vocab)
    att_encdec = ABED(args.vocab, args.hidden_size, args.maxout_hidden_size, args.embed_size)
    if args.use_gpu:
        att_encdec.to_gpu()
    if args.source_validation:
        if os.path.exists(PLOT_DIR)==False: os.mkdir(PLOT_DIR)
        fp_loss = open(PLOT_DIR+"loss", "w")
        fp_loss_val = open(PLOT_DIR+"loss_val", "w")

    opt = optimizers.AdaDelta(args.rho, args.eps)
    opt.setup(att_encdec)
    opt.add_hook(optimizer.WeightDecay(DECAY_COEFF))
    opt.add_hook(optimizer.GradientClipping(CLIP_THR))
    for epoch in xrange(args.epochs):
        print "--- epoch: %s/%s ---"%(epoch+1, args.epochs)
        source_gen = word_list(args.source)
        target_gen = word_list(args.target)
        batch_gen = batch(sort(source_gen, target_gen, 100*args.minibatch), args.minibatch)
        n = 0
        total_loss = 0.0
        for source_batch, target_batch in batch_gen:
            n += len(source_batch)
            source_batch = fill_batch_end(source_batch)
            target_batch = fill_batch_end(target_batch)
            hyp_batch, loss = forward(source_batch, target_batch, source_vocab, target_vocab, att_encdec, True, 0)
            total_loss += loss.data*len(source_batch)
            closed_test(source_batch, target_batch, hyp_batch)

            loss.backward()
            opt.update()
            print "[n=%s]"%(n)
        print "[total=%s]"%(n)
        prefix = args.model_path + '%s'%(epoch+1)
        serializers.save_hdf5(prefix+'.attencdec', att_encdec)
        if args.source_validation:
            total_loss_val, n_val = validation_test(args, att_encdec, source_vocab, target_vocab)
            fp_loss.write("\t".join([str(epoch), str(total_loss/n)+"\n"]))
            fp_loss_val.write("\t".join([str(epoch), str(total_loss_val/n_val)+"\n"])) 
            fp_loss.flush()
            fp_loss_val.flush()
        hyp_params = att_encdec.get_hyper_params()
        Backup.dump(hyp_params, args.model_path+HPARAM_NAME)
        source_vocab.save(args.model_path+SRC_VOCAB_NAME)
        target_vocab.save(args.model_path+TAR_VOCAB_NAME)
    hyp_params = att_encdec.get_hyper_params()
    Backup.dump(hyp_params, args.model_path+HPARAM_NAME)
    source_vocab.save(args.model_path+SRC_VOCAB_NAME)
    target_vocab.save(args.model_path+TAR_VOCAB_NAME)
    if args.source_validation:
        fp_loss.close()
        fp_loss_val.close()

def show(src, tar, hyp, t):
    print '----- %s -----'%(t)
    print 'source: %s'%(' '.join([w for w in src]))
    print 'target: %s'%(' '.join([w for w in tar]))
    print 'hyp: %s'%(' '.join([w for w in hyp]))
    """
    try:
        print 'SENTENCE BLEU: %s'%(sentence_bleu([tar], hyp))
    except ZeroDivisionError:
        print 'SENTENCE BLEU: 0.0'
    """
    print '--------------'

def fwrite(src, tar, hyp, fp):
    fp.write('----------\n')
    fp.write('source: %s\n'%(' '.join([w for w in src])))
    fp.write('target: %s\n'%(' '.join([w for w in tar])))
    fp.write('hyp: %s\n'%(' '.join([w for w in hyp])))
    """
    try:
        fp.write('SENTENCE BLEU: %s\n'%(sentence_bleu([tar], hyp)))
    except ZeroDivisionError:
        fp.write('SENTENCE BLEU: 0.0\n')
    """
    fp.write('--------------\n')
    fp.flush()

def closed_test(src_batch, tar_batch, hyp_batch):
    for k in range(len(src_batch)):
        hyp_batch[k].append(END)
        hyp = hyp_batch[k][:hyp_batch[k].index(END)]
        tar = tar_batch[k][:tar_batch[k].index(END)]
        show(src_batch[k], tar, hyp, "CLOSED")

def validation_test(args, encdec, src_vocab, tar_vocab):
    src_gen = word_list(args.source_validation)
    tar_gen = word_list(args.target_validation)
    batch_gen = batch(sort(src_gen, tar_gen, 100*args.minibatch), args.minibatch)
    total_loss = 0.0
    n = 0
    for src_batch, tar_batch in batch_gen:
        n += len(src_batch)
        src_batch= fill_batch_end(src_batch)
        tar_batch = fill_batch_end(tar_batch)
        hyp_batch, loss = forward(src_batch, tar_batch, src_vocab, tar_vocab, encdec, True, 0)
        total_loss += loss.data*len(src_batch)
        for i, hyp in enumerate(hyp_batch):
            hyp.append(END)
            hyp = hyp[:hyp.index(END)]
            tar = tar_batch[i][:tar_batch[i].index(END)]
            show(src_batch[i], tar, hyp, "VALIDATION")
    return total_loss, n

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
            source_batch = fill_batch_end(source_batch)
            target_batch = fill_batch_end(target_batch) 
            if args.beam_search:
                hyp_batch = forward_beam(source_batch, None, source_vocab, target_vocab, att_encdec, False, args.limit, args.beam_size)
            else:
                hyp_batch = forward(source_batch, None, source_vocab, target_vocab, att_encdec, False, args.limit)
            for i, hyp in enumerate(hyp_batch):
                hyp.append(END)
                hyp = hyp[:hyp.index(END)]
                show(source_batch[i], target_batch[i], hyp, "TEST")
                fwrite(source_batch[i], target_batch[i], hyp, fp)

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
    DEF_LIMIT = 20
    DEF_BEAM = 2

    p = argparse.ArgumentParser(
        description = "A Neural Attention Model for Machine Translation."
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
            "-source_validation",
            type=str,
            help="path_to_validation_source, if set this option, validation test will be conducted in training."
    )
    p.add_argument(
            "-target_validation",
            type=str,
            help="path_to_validation_target, if set this option, validation test will be conducted in training."
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
            "--beam_search",
            action="store_true",
            help="conduct beam search in decoding"
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
    p.add_argument(
            "-beam_size",
            type=int,
            default=DEF_BEAM,
            help="beam size of beam search"
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
