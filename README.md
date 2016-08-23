This is an implementation of the Neural Machine Translation model with

	- [Bahdanau et al. 2015] (Original) Global Attention : https://arxiv.org/abs/1409.0473

	- [Luong et al. 2015] (Simple) Global Attention : http://www.aclweb.org/anthology/D15-1166

        - [Luong et al. 2015] Local Attention : http://www.aclweb.org/anthology/D15-1166

The implementation uses LSTM as gated hidden unit instead of GRU to implement [Bahdanau et al. 2015].

This is based on the https://github.com/odashi/chainer_examples.
Thank you.

Environment:
* python 2.7
* Anaconda
* Chainer 1.8.2
