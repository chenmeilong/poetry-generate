#!/usr/bin/python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import gluon as g


class CharRNN(g.Block):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers,
                 dropout):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        with self.name_scope():
            self.word_to_vec = g.nn.Embedding(num_classes, embed_dim)
            self.rnn = g.rnn.GRU(hidden_size, num_layers, dropout=dropout)
            self.proj = g.nn.Dense(num_classes)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = nd.zeros(
                (self.num_layers, batch, self.hidden_size), ctx=mx.gpu())
        word_embed = self.word_to_vec(x)  # batch x len x embed
        word_embed = word_embed.transpose((1, 0, 2))  # len x batch x embed
        out, h0 = self.rnn(word_embed, hs)  # len x batch x hidden
        le, mb, hd = out.shape
        out = out.reshape((le * mb, hd))
        out = self.proj(out)
        out = out.reshape((le, mb, -1))
        out = out.transpose((1, 0, 2))  # batch x len x hidden
        return out.reshape((-1, out.shape[2])), h0
