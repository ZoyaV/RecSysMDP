from typing import Sequence, Optional

import d3rlpy
import torch
from d3rlpy.models.torch import EncoderWithAction, VectorEncoder
from d3rlpy.models.torch.encoders import _VectorEncoder
from torch import nn
from torch.nn import functional as F


class VectorEncoderWithAction(_VectorEncoder, EncoderWithAction):
    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = (observation_shape[0] + action_size,)
        super().__init__(
            observation_shape=concat_shape,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        #raise Exception(x.shape, action.shape)
        try:
            x = torch.cat([x, action], dim=1)
        except:
            one_hot = F.one_hot(action.to(torch.int64).view(-1), num_classes=self.action_size)
            x = torch.cat([x, one_hot], dim=1)
            #raise Exception(one_hot.shape, one_hot)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) -> int:
        return self._action_size


class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return VectorEncoder(observation_shape, [self.feature_size])

    def create_with_action(self, observation_shape, action_size):
        return VectorEncoderWithAction(observation_shape, action_size, [self.feature_size])

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
