import torch
import torch.nn as nn
from itertools import tee
from collections import OrderedDict


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.cfg = cfg
        _configs = self.cfg.MODEL.FC.LAYERS
        self._layers = []
        in_size = self.cfg.MODEL.FC.INPUT
        for i, ff_config in enumerate(_configs):
            if ff_config['norm']:
                batchnorm_layer = nn.BatchNorm1d(in_size)
                self._layers.append(batchnorm_layer)
            linear_layer = nn.Linear(in_size, ff_config['to_size'], bias=True)
            activation_layer = nn.ELU()
            self._layers.append(linear_layer)
            self._layers.append(activation_layer)
            if ff_config['dropout']:
                dropout = nn.Dropout(p=ff_config['dropout'])
                self._layers.append(dropout)
            in_size = ff_config['to_size']

        self.sequential = nn.Sequential(*self._layers)

    def forward(self, x):
        # out = x
        # for i in range(len(self._sizes) - 1):
        #     fc = self.__getattr__('linear_layer_' + str(i))
        #     ac = self.__getattr__('activation_layer_' + str(i))
        #     out = ac(fc(out))
        out = self.sequential(x)
        return out
