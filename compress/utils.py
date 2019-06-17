#-*- coding: utf-8 -*-

from mxnet.gluon import nn

__all__ = ['collect_conv_and_fc']
__author__ = 'YaHei'


def collect_conv_and_fc(net, exclude=[]):
    collector = []
    def _collect(m):
        if type(m) in (nn.Conv2D, nn.Dense) and m not in exclude:
            collector.append(m)
    net.apply(_collect)

    return collector