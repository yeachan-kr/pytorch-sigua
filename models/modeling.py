""" Modeling layer Implementation """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def call_bn(bn, x):
    return bn(x)


class NoiseModel(nn.Module):
    def __init__(self, num_class: int):
        super(NoiseModel, self).__init__()
        self.num_class = num_class
        self.transition_mat = Parameter(torch.eye(num_class))

    def forward(self, x):
        """
        x:
            shape = (batch, num_class) (probability distribution)
        return:
            noise distribution
        """
        out = torch.matmul(x, self.transition_mat)
        return out


class TextCNN(nn.Module):
    def __init__(self,  vocab: dict, num_class: int, drop_rate: float, pre_weight: np.ndarray = None):
        super(TextCNN, self).__init__()
        self.emb_dim = 300
        self.num_filters = [100, 100, 100]
        self.size_filters = [3, 4, 5]

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=self.emb_dim,
                                      padding_idx=vocab['<pad>'])

        if pre_weight is not None:
            self.embedding.from_pretrained(pre_weight)

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n, kernel_size=(k, self.emb_dim))
                                          for n, k in zip(self.num_filters, self.size_filters)])
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight.data)

        num_features = sum(self.num_filters)
        self.drop_out = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_class)
        # nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):

        # 1. Embedding layer
        out = self.embedding(x).unsqueeze(1)

        # 2. 1-D convolution layer
        features = []
        for conv in self.conv_layers:
            h = conv(out).squeeze()
            h = F.relu(h)
            h = F.max_pool1d(input=h, kernel_size=h.size(-1)).squeeze()
            features.append(h)
        features = torch.cat(features, dim=-1)

        # 3. two-layered linear layers (with dropout)
        features = self.fc1(features)
        features = F.relu(features)
        features = self.drop_out(features)
        out = self.fc2(features)
        return out


class CNN(nn.Module):
    def __init__(self, num_class, dropout_rate):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        h = self.c1(x)
        h = F.relu(h)
        h = self.c2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)

        h = self.dropout1(h)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.dropout2(h)
        logit = self.fc2(h)
        return logit
