import torch.nn as nn
import torch
import torch.nn.functional as F
from IPython import embed

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_block_1d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Registrator(nn.Module):
    def __init__(self):
        super(Registrator, self).__init__()
        self.fc_params_support = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)
        self.fc_params_query = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)

    def forward(self, support_set, query_set):
        support_set_2 = self.fc_params_support(support_set)
        query_set_2 = self.fc_params_query(query_set)
        return support_set_2, query_set_2
