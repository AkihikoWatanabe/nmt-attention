# coding=utf-8

import matplotlib.pyplot as plt
import argparse

def load(path):
    with open(path) as fp:
        dic = dict((float(l.split()[0]), float(l.split()[1])) for l in fp.read().strip().split("\n"))
    return dic

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('plot_path')
    args = p.parse_args()

    return args

if __name__=="__main__":
    args = parse_args()
    loss = load("%s/loss"%args.plot_path)
    loss_val = load("%s/loss_val"%args.plot_path)
    
    plt.plot(loss.keys(), loss.values(), color='yellow', linewidth=2.0, label='loss')
    plt.plot(loss_val.keys(), loss_val.values(), color='red', linewidth=2.0, label='loss_validation')
    plt.legend()
    plt.show()
