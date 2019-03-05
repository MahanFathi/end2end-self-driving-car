import torch
import torch.nn as nn
from model.layer.feed_forward import FeedForward


class PilotNet(nn.Module):
    def __init__(self, cfg, mode):
        super(PilotNet, self).__init__()
        self.cfg = cfg
        self.training = mode == 'training'

        # BUILD CNN BACKBONE
        cnn_layers = []
        input_channels = self.cfg.MODEL.CNN.INPUT_CHANNELS
        cnn_configs = self.cfg.MODEL.CNN.LAYERS
        for cnn_config in cnn_configs:
            cnn_layer = []
            cnn_layer.append(nn.Conv2d(input_channels,
                                       cnn_config['out_channels'],
                                       cnn_config['kernel'],
                                       cnn_config['stride']),
                             )
            if cnn_config['pool_kernel']:
                cnn_layer.append(nn.AvgPool2d(cnn_config['pool_kernel']))
            cnn_layer.append(nn.ELU())
            cnn_layer.append(nn.Dropout2d(p=self.cfg.MODEL.CNN.DROPOUT))
            input_channels = cnn_config['out_channels']
            cnn_layers.extend(cnn_layer)

        self.cnn_backbone = nn.Sequential(cnn_layers)

        # BUILD FULLY CONNECTED
        self.embedding = FeedForward(self.cfg)
        last_embedding_size = self.cfg.MODEL.FC.LAYERS[-1]['to_size']
        self.to_out = nn.Linear(last_embedding_size, 1)
        self.feed_forward = nn.Sequential([self.embedding, self.to_out])

        # BUILD LOSS CRITERION
        self.loss_criterion = nn.MSELoss()

    def forward(self, input, targets=None):
        batch_size = input.size(0)
        normalized_input = input / 127.5 - 1
        cnn_features = self.cnn_backbone(normalized_input)
        predictions = self.feed_forward(cnn_features.view([batch_size, -1]))

        if self.training:
            assert targets is not None
            loss = self.loss_criterion(targets, predictions)
            return predictions, loss

        return predictions
