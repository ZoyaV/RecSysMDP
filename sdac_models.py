from typing import Sequence, Optional, List
import torch
from torch import nn
from torch.nn import functional as F
from d3rlpy.models.torch import EncoderWithAction, Encoder
import d3rlpy
import numpy as np

class _PixelEncoderAboba(nn.Module):  # type: ignore

    _observation_shape: Sequence[int]
    _feature_size: int
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _activation: nn.Module
    _convs: nn.ModuleList
    _conv_bns: nn.ModuleList
    _fc: nn.Linear
    _fc_bn: nn.BatchNorm1d
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512*10,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
       # print("abobe!!!")
        # default architecture is based on Nature DQN paper.
       
        if feature_size is None:
            feature_size = 64
        hidden_units = [feature_size,feature_size]
        
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = feature_size
        self._activation = activation
        self._use_dense = False
        use_dense = self._use_dense = False
       # print(observation_shape)
        if len(observation_shape)>1:
            inp = observation_shape[1]*observation_shape[2]
        else:
            inp = observation_shape[0]#* observation_shape[1]*observation_shape[2]
        in_units = [inp] + list(hidden_units[:-1])
      #  print("!!!!!!!")
      #  print(in_units)
      #  print("!!!!!!!!")
       # print("oOoOooooOoOo", in_units)
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))


    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
       # x = 
     #   print("--------")
      #  print(x.shape)
        x = x.reshape(x.shape[0],-1)
        h = x
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) -> nn.Linear:
        return self._fcs[-1]

class PixelEncoderAboba(_PixelEncoderAboba, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # print("lololo ---->")
       # print(x)
       # print("lololo ---->")
        x = x[:,0, :,:]
        x = x /255
        h = self._fc_encode(x)
#         if self._use_batch_norm:
#             h = self._bns[-1](h)
#         if self._dropout_rate is not None:
#             h = self._dropouts[-1](h)
      #  print("--------")
       # print(h)
       # print(h.shape)
       # print("--------")
        return h
    

class PixelEncoderWithActionAboba(_PixelEncoderAboba, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512*10,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        item_mapping = None
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        obs_shape = observation_shape[1]*observation_shape[2]
        concat_shape = (obs_shape + 8,)#action_size,)
        super().__init__(
            observation_shape=concat_shape,
            filters=filters,
            feature_size=feature_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        print("---------")
        print(self._action_size)
        print("---------")
   #     exit()
        self.item_mapping = item_mapping
        #self.action_encoder = nn.Embedding(self._action_size, 8)

    def _get_linear_input_size(self) -> int:
        size = super()._get_linear_input_size()
        return size + 8#self._action_size

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        #print("lololo ---->")
        #print(x[:,:3, :,:])
       # print("lololo ---->")
        x = x[:,0, :,:] /255
      # print(x)
        x = x.reshape(x.shape[0],-1)    
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        if len(action.shape)>1:
           # print(action.shape)
           # action = self.action_encoder(action)
           # print("___________________")
          #  print(action.shape)
            action = torch.argmax(action, axis = 1)
           # print(action.shape)
            action = np.asarray(list(map(lambda x: self.item_mapping[x.item()], action)))
            action = torch.from_numpy(action).cuda().float()
           # print(action.shape)
           # print("___________________")
            x = torch.cat([x, action], dim=1)
        else:
           # print("----------------------------")
           # print(action.shape)
           # print(action[0])
            action = np.asarray(list(map(lambda x:self.item_mapping[x.item()], action)))
            action = torch.from_numpy(action).cuda().float()
           # print(action.shape)
           # print("----------------------------")
            #one_hot = F.one_hot(action.to(torch.int64).view(-1), num_classes=self.action_size).float()
            #one_hot = self.action_encoder(one_hot)
            x = torch.cat([x, action], dim=1)
      #  print(x.shape)
        h = self._fc_encode(x)
#         if self._use_batch_norm:
#             h = self._bns[-1](h)
#         if self._dropout_rate is not None:
#             h = self._dropouts[-1](h)
       #print("+++++++++")
       #print(h)
     #   print(h.shape)
        #rint("+++++++++")
        return h

    @property
    def action_size(self) -> int:
        return self._action_size


class CustomEncoderFactorySDAC(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size, item_mapping):
        self.feature_size = feature_size
        self.item_mapping = item_mapping

    def create(self, observation_shape):
        return PixelEncoderAboba(observation_shape)

    def create_with_action(self, observation_shape, action_size):
        return PixelEncoderWithActionAboba(
            observation_shape, action_size, item_mapping=self.item_mapping 
        )

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}