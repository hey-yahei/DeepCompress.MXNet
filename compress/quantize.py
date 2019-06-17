#-*- coding: utf-8 -*-

import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from sklearn.cluster import KMeans
from .utils import collect_conv_and_fc

__all__ = ['CodebookQuantizer', 'LinearRangeQuantizer']
__author__ = 'YaHei'


class CodebookQuantizer(object):
    def __init__(self, net, exclude=[], default_bits=5, bits={}):
        self._net = net
        self._blocks = {m: default_bits for m in collect_conv_and_fc(net, exclude)}
        self._blocks.update(bits)

    def _quant_codebook(self, data, bits):
        shape = data.shape
        min_ = data.min()
        max_ = data.max()
        init_space = np.linspace(min_, max_, 2**bits)
        kmeans = KMeans(n_clusters=len(init_space),
                        init=init_space.reshape(-1,1),
                        n_init=1,
                        precompute_distances=True,
                        n_jobs=-1,   # all processors
                        algorithm="full")
        kmeans.fit(data.reshape(-1,1))
        quant_data = kmeans.labels_.reshape(shape)
        cluster_center = kmeans.cluster_centers_
        real_data = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
        return quant_data, cluster_center, real_data

    def apply_quantize(self, set_zero=False):
        for blk, bits in self._blocks.items():
            weight_ctx = blk.weight.list_ctx()[0]
            weight = blk.weight.data().asnumpy()
            print(f'Quantize {blk.name} with {weight.size} parameters into uint{bits}...')
            quant_weight, cluster_center, real_weight = self._quant_codebook(weight, bits)

            if set_zero:
                label = np.argmin(abs(cluster_center))
                nearest = cluster_center[label]
                real_weight[real_weight == nearest] = 0
                cluster_center[label] = 0
                blk.zero_label = label

            blk.weight.set_data(real_weight)
            blk.quant_weight = quant_weight
            blk.cluster_center = nd.array(cluster_center, ctx=weight_ctx)

            masks = []
            for i in range(2 ** self._blocks[blk]):
                mask = nd.array(np.where(quant_weight == i, 1, 0), ctx=weight_ctx)
                masks.append(mask)
            blk.masks = nd.stack(*masks)

    def update(self, lr):
        for blk, bits in self._blocks.items():
            wt = blk.weight.data()
            wt_grad = blk.weight.grad()
            for i in range(2 ** bits):
                if i == blk.zero_label:
                    continue
                delta_cluster_center = (wt_grad * blk.masks[i]).sum() * lr
                blk.cluster_center[i] -= delta_cluster_center
                wt -= delta_cluster_center * blk.masks[i]


class LinearRangeQuantizer(object):
    def __init__(self, net, all_conv=True, all_fc=False, default_bits=8):
        self._net = net
        self._quant_blks = {m: default_bits for m in self._collect_conv_and_fc(all_conv, all_fc)}

    def _collect_conv_and_fc(self, all_conv, all_fc):
        type_lst = []
        if all_conv:  type_lst.append(nn.Conv2D)
        if all_fc:    type_lst.append(nn.Dense)
        modules = []
        def _collect(m):
            if type(m) in type_lst:
                modules.append(m)
        self._net.apply(_collect)
        return modules

    def set_quant_blocks(self, quant_blks):
        self._quant_blks = quant_blks

    def _quant_asymmetric(self, data, bits):
        min_ = data.min()
        max_ = data.max()
        scale = (max_ - min_) / (2 ** bits - 1)
        quant_data = np.round( (data - min_) / scale )
        real_data = quant_data * scale + min_
        return quant_data, (min_, scale), real_data

    def _quant_per_channels(self, weight, bits):
        n_channel = weight.shape[0]
        quant_weight = np.zeros_like(weight)
        real_weight = np.zeros_like(weight)
        mins = []
        scales = []
        for i in range(n_channel):
            filter = weight[i]    # keep dim
            quant_filter, (min_, scale), real_filter = self._quant_asymmetric(filter, bits)
            quant_weight[i] = quant_filter
            real_weight[i] = real_filter
            mins.append(min_)
            scales.append(scale)
        return quant_weight, (mins, scales), real_weight

    def apply_quantize(self, per_channel=True):
        quant_func = self._quant_per_channels if per_channel else self._quant_asymmetric
        def _apply_quantize(m):
            if m in self._quant_blks:
                weight = m.weight.data().asnumpy()
                n_bits = self._quant_blks[m]
                quant_weight, (min_, scale), real_weight = quant_func(weight, n_bits)
                m.weight.set_data(real_weight)
                m.quant_weight = quant_weight
                m.min_ = min_
                m.scale = scale
        self._net.apply(_apply_quantize)

