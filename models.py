import d3rlpy
from torch import nn
import torch
import numpy as np


class EncoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def get_encoder_out_size(self):
        return self.encoder_out_size

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        """Default implementation, can be overridden in derived classes."""
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def forward_fc_blocks(self, x):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x)

        return x


class HistoryEncoder(EncoderBase):
    def __init__(self, observation_shape,  feature_size):
        super().__init__()
        self.feature_size = feature_size
        obs_shape = observation_shape
        self.input_size = obs_shape[-1]* obs_shape[-2]

        self.good_encoder = nn.Linear(self.input_size, self.feature_size)
        self.bad_encoder = nn.Linear(self.input_size, self.feature_size)
        self.hist_encoder = nn.Linear(self.input_size, self.feature_size)

        self.out = nn.Linear(2*self.feature_size, self.feature_size)

    def forward(self, obs_dict):
        print("Aboba!")
        exit()
        hist_emb = self.hist_encoder(obs_dict[:,0,:,:].view(-1,self.input_size))
        good_emb = self.good_encoder(obs_dict[:,1,:,:].view(-1,self.input_size))
        x = torch.cat([good_emb, hist_emb], axis =  -1)
        x = self.out(x)
        return x

    def get_feature_size(self):
        return self.feature_size




class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"  # this is necessary

    def __init__(self,  feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return HistoryEncoder(observation_shape, self.feature_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}