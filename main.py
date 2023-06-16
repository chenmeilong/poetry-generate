#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon as g
import mxnet.autograd as ag
from CharRNN import CharRNN
from data_utils import TextConverter, TextData


def train_epoch(ctx, model, dataloader, criterion, optimizer, clip):
    running_loss = 0.0
    n_total = 0.0
    for batch in dataloader:
        x, y = batch
        y = y.astype('float32').as_in_context(ctx).swapaxes(0, 1)
        x = x.as_in_context(ctx).swapaxes(0, 1)
        mb_size = x.shape[0]
        with ag.record():
            out = model(x)
            batch_loss = criterion(out, y)
        batch_loss.backward()
        grads = [i.grad(ctx) for i in model.collect_params().values()]
        total_norm = g.utils.clip_global_norm(grads, clip * y.shape[0] * y.shape[1])

        if np.isfinite(total_norm):
            optimizer.step(mb_size)
            running_loss += nd.sum(batch_loss).asscalar()
            n_total += mb_size
        else:
            raise UserWarning('nan/inf detected. skipping batch')
    return running_loss / n_total


def train(ctx, n_epoch, model, dataloader, optimizer, criterion, clip):
    for e in range(n_epoch):
        print('{}/{}'.format(e + 1, n_epoch))
        since = time.time()
        loss = train_epoch(ctx, model, dataloader, criterion, optimizer, clip)
        print('Loss: {:.6f}, Time: {:.3} s'.format(loss, time.time() - since))
        if (e + 1) % 1000 == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            model.save_params('./checkpoints/model_{}.params'.format(e + 1))


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = nd.topk(preds, axis=2, k=top_n, ret_typ='both')
    top_pred_label = top_pred_label.asnumpy()
    top_pred_prob /= nd.sum(top_pred_prob, axis=2, keepdims=True)
    top_pred_prob = top_pred_prob.asnumpy().reshape((-1, ))
    top_pred_label = top_pred_label.reshape((-1, ))
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


def sample(ctx, model, checkpoint, convert, arr_to_text, prime, text_len=20):
    '''
    将载入好权重的模型读入，指定开始字符和长度进行生成，将生成的结果保存到txt文件中
    checkpoint: 载入的模型
    convert: 文本和下标转换
    prime: 起始文本
    text_len: 生成文本长度
    '''
    model.load_params(checkpoint, ctx=ctx)
    samples = [convert(c) for c in prime]
    input_txt = nd.array(samples).reshape((-1 ,1)).as_in_context(ctx)
    embed = model[0](input_txt)
    hs = nd.zeros(model[1].state_info(1)[0]['shape'], ctx=ctx)
    _, init_state = model[1](embed, hs)

    result = samples
    model_input = input_txt[:, input_txt.shape[1] - 1].reshape((-1, 1))
    for i in range(text_len):
        # out是输出的字符，大小为1 x vocab
        # init_state是RNN传递的hidden state
        with mx.autograd.predict_mode():
            embed = model[0](model_input)
            out, init_state = model[1](embed, init_state)
            out = model[2](out)
        pred = pick_top_n(out)
        model_input = nd.array(pred).reshape((-1, 1)).as_in_context(ctx)
        result.append(pred[0])
    return arr_to_text(result)


def main():
    '''main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True, help='训练还是预测, train or eval')
    parser.add_argument('--txt', required=True, help='进行训练的txt文件')
    parser.add_argument('--batch', default=128, type=int, help='训练的batch size')
    parser.add_argument('--epoch', default=5000, type=int, help='跑多少个epoch')
    parser.add_argument('--len', default=100, type=int, help='输入模型的序列长度')
    parser.add_argument(
        '--max_vocab', default=5000, type=int, help='最多存储的字符数目')
    parser.add_argument('--embed', default=512, type=int, help='词向量的维度')
    parser.add_argument('--hidden', default=512, type=int, help='RNN的输出维度')
    parser.add_argument('--n_layer', default=2, type=int, help='RNN的层数')
    parser.add_argument(
        '--dropout', default=0.5, type=float, help='RNN中drop的概率')
    parser.add_argument('--begin', default='我', help='给出生成文本的开始')
    parser.add_argument('--pred_len', default=20, type=int, help='生成文本的长度')
    parser.add_argument('--checkpoint', help='载入模型的位置')
    parser.add_argument('--clip', default=0.2, type=float, help='权重上限')
    parser.add_argument('--use-gpu', default=False, help='是否使用的GPU')
    opt = parser.parse_args()
    print(opt)

    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    model = g.nn.Sequential()
    with model.name_scope():
        model.add(g.nn.Embedding(convert.vocab_size, opt.embed))
        model.add(g.rnn.GRU(opt.hidden, opt.n_layer, dropout=opt.dropout))
        model.add(g.nn.Dense(convert.vocab_size, flatten=False))

    ctx = mx.gpu(0) if opt.use_gpu else mx.cpu()
    model.initialize(ctx=ctx)

    if opt.state == 'train':
        dataset = TextData(opt.txt, opt.len, convert.text_to_arr)
        dataloader = g.data.DataLoader(dataset, opt.batch, shuffle=True)
        lr_sch = mx.lr_scheduler.FactorScheduler(
            int(1000 * len(dataloader)), factor=0.1)
        optimizer = g.Trainer(model.collect_params(), 'adam', {
            'learning_rate': 1e-3,
            'clip_gradient': 3,
            'lr_scheduler': lr_sch
        })
        cross_entropy = g.loss.SoftmaxCrossEntropyLoss()
        train(ctx, opt.epoch, model, dataloader, optimizer, cross_entropy, opt.clip)

    elif opt.state == 'eval':
        pred_text = sample(ctx, model, opt.checkpoint, convert.word_to_int,
                           convert.arr_to_text, opt.begin, opt.pred_len)
        print(pred_text)
        with open('./generate.txt', 'a') as f:
            f.write(pred_text)
            f.write('\n')
    else:
        print('Error state, must choose from train and eval!')


if __name__ == '__main__':
    main()
