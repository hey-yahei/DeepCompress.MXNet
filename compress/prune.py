#-*- coding: utf-8 -*-

from mxnet.gluon import nn
import types
import numpy as np
from .utils import collect_conv_and_fc

__all__ = ['Pruner']
__author__ = 'YaHei'


def prune(m, threshold):
    mask = m.mask.data().asnumpy()
    weight = m.weight.data().asnumpy() * mask
    new_mask = np.where(abs(weight) < threshold, 0, mask)
    m.weight.set_data(weight * new_mask)
    m.mask.set_data(new_mask)

class Pruner(object):
    def __init__(self, net, exclude=[]):
        self._net = net
        self._blocks = collect_conv_and_fc(net, exclude)

    def convert(self):
        def _forward(self, F, x, weight, mask, bias=None):
            return self.old_forward(F, x, weight * mask, bias)

        for blk in self._blocks:
            weight_ctx = blk.weight.list_ctx()[0]
            blk.mask = blk.params.get("mask",
                                       shape=blk.weight.shape, init="ones",
                                       allow_deferred_init=True,
                                       differentiable=False)
            blk.mask.initialize(ctx=weight_ctx)
            # blk.mask = nd.ones_like(blk.weight.data(), ctx=weight_ctx)
            blk.old_forward = blk.hybrid_forward
            blk.hybrid_forward = types.MethodType(_forward, blk)

    def apply(self):
        for blk in self._blocks:
            weight = blk.weight.data()
            blk.weight.set_data(weight * blk.mask)

    def prune_by_std(self, s=0.25, s_factor={}):
        for blk in self._blocks:
            weight = blk.weight.data().asnumpy()
            threshold = np.std(weight) * s * s_factor.get(blk, 1.)
            prune(blk, threshold)

    def prune_by_percent(self, q=5.0):
        for blk in self._blocks:
            weight = blk.weight.data().asnumpy()
            alive = weight[np.nonzero(weight)]
            percentile_value = np.percentile(abs(alive), q)
            prune(blk, percentile_value)

    def get_sparsity(self):
        sparsity = {}
        for blk in self._blocks:
            mask = blk.mask.data().asnumpy()
            sparsity[blk.name] = (mask == 0).mean()
        return sparsity

    def get_weight_volume(self):
        counter = 0
        for blk in self._blocks:
            mask = blk.mask.data().asnumpy()
            counter += mask.sum()
        return counter

    def apply_mask_to_grad(self):
        for blk in self._blocks:
            mask = blk.mask.data()
            wt_grad = blk.weight.grad()
            wt_grad *= mask

    def apply_pruning(self):
        for blk in self._blocks:
            weight = blk.weight.data()
            mask = blk.mask.data()
            blk.weight.set_data(weight * mask)
