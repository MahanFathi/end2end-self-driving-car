import torch
import torch.nn as nn


class BackwardPilotNet(nn.Module):
    def __init__(self, cfg):
        super(BackwardPilotNet, self).__init__()
        self.cfg = cfg

        # BUILD CNN BACKBONE
        cnn_layers = []
        input_channels = self.cfg.MODEL.BACKWARD_CNN.INPUT_CHANNELS
        cnn_configs = self.cfg.MODEL.BACKWARD_CNN.LAYERS
        for cnn_config in cnn_configs:
            cnn_layer = nn.ConvTranspose2d(input_channels,
                                           cnn_config['out_channels'],
                                           cnn_config['kernel'],
                                           stride=cnn_config['stride'])

            input_channels = cnn_config['out_channels']
            cnn_layers.append(cnn_layer)

        self.backward_layers = cnn_layers

        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, activations, targets=None):

        last_activation = torch.ones_like(activations[-1])
        for back_op, activation in zip(self.backward_layers, reversed(activations)):
            summation = torch.mul(back_op(last_activation), activation)
            last_activation = summation
            last_activation = self.normalization(last_activation)
        return last_activation

    @staticmethod
    def normalization(tensor):
        omin = tensor.min(2, keepdim=True)[0].min(3, keepdim=True)[0].mul(-1)
        omax = tensor.max(2, keepdim=True)[0].max(3, keepdim=True)[0].add(omin)
        tensor = torch.add(tensor, omin.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
        tensor = torch.div(tensor, omax.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
        return tensor

    # TODO ??
    # def normalize(self, img, min, max):
    #     img.clamp_(min=min, max=max)
    #     img.add_(-min).div_(max - min + 1e-5)
