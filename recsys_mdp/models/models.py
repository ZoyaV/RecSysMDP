from __future__ import annotations

from typing import Sequence, ClassVar

import numpy as np
import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch import Encoder, EncoderWithAction

from recsys_mdp.mdp.base import USER_ID_COL, ITEM_ID_COL
from recsys_mdp.utils.run.config import extracted


class ObservationComponent:
    name: str
    start: int
    end: int
    encoder: nn.Module | None

    def __init__(self, name: str, indices_range: tuple[int, int], encoder: nn.Module = None):
        self.name = name
        self.start, self.end = indices_range
        self.encoder = encoder

    def encode(self, x: torch.Tensor, take_slice: bool = True) -> torch.Tensor:
        if take_slice:
            # slice over the last dim to skip batch dimension
            x = x[..., self.start:self.end]
        if self.encoder is None:
            return x
        return self.encoder(x)

    @property
    def output_dim(self):
        return 1 if self.encoder is None else self.encoder.output_dim

    @property
    def total_output(self):
        return self.output_dim * (self.end - self.start)


class StateEncoder(nn.Module, Encoder):
    def __init__(
            self, *,
            observation_components: dict[str, ObservationComponent],
            observation_encoder: str,
            hidden_dim, output_dim, **state_encoding_config
    ):
        super().__init__()
        self.output_dim = output_dim
        self.observation_components, self.user_component = extracted(
            observation_components, USER_ID_COL
        )
        self.net = nn.Sequential(
            resolve_observation_encoder(observation_encoder, **state_encoding_config),
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.ReLU(),
        )
        # FIXME: test initialization
        # self.initialize()

    def initialize(self):
        """weight init"""
        for params in self.net.parameters():
            nn.init.kaiming_uniform_(params, nonlinearity='relu')

    def forward(self, x: torch.Tensor):
        # x shape: (batch, flatten_raw_observation) float

        user = None
        if self.user_component is not None:
            # (batch, user_ids=1, user_embedding)
            user = self.user_component.encode(x)
            # remove `user_ids` dim
            user = torch.flatten(user, start_dim=1)

        # each: (batch, interaction_timestep, part_embedding_size)
        interaction_history_by_components = [
            component.encode(x)
            for component in self.observation_components.values()
        ]
        interaction_history = torch.cat(interaction_history_by_components, dim=2)

        # have to pass a single object as it's required by Sequential module
        pair = user, interaction_history
        return self.net(pair)

    def get_feature_size(self):
        return self.output_dim


class ActorEncoderWithAction(nn.Module, EncoderWithAction):
    def __init__(
            self, *, action_size,
            observation_components: dict[str, ObservationComponent],
            state_encoding_method: str,
            hidden_dim, output_dim, **state_encoding_config
    ):
        super().__init__()
        assert action_size == 1, f'A single integer action (=item_id) expected, got {action_size}!'

        self.state_encoder = StateEncoder(
            observation_components=observation_components,
            observation_encoder=state_encoding_method,
            hidden_dim=hidden_dim, output_dim=output_dim,
            **state_encoding_config
        )

        self.output_dim = self.state_encoder.output_dim
        self.action_component = observation_components.get(ITEM_ID_COL, None)

        action_size = self.action_component.output_dim
        self.net = nn.Sequential(
            nn.Linear(self.output_dim + action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.ReLU(),
        )
        self.initialize()

    def initialize(self):
        """weight init"""
        for params in self.net.parameters():
            nn.init.kaiming_uniform_(params, nonlinearity='relu')

    def forward(self, x, action):
        state = self.state_encoder(x)
        action = self.action_component.encode(action, take_slice=False)
        state_action = torch.cat([state, action], dim=1)
        return self.net(state_action)

    def get_feature_size(self):
        return self.output_dim


class ActorEncoderFactory(EncoderFactory):
    TYPE: ClassVar[str] = "custom"

    def __init__(
            self, *,
            observation_components: dict[str, ObservationComponent],
            state_encoding_method: str,
            hidden_dim, output_dim,
            **state_encoding_config
    ):
        self.observation_components = observation_components
        self.state_encoding_method = state_encoding_method
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.state_encoding_config = state_encoding_config

    def create(self, observation_shape: Sequence[int]) -> Encoder:
        return StateEncoder(
            observation_components=self.observation_components,
            observation_encoder=self.state_encoding_method,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            **self.state_encoding_config
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        return ActorEncoderWithAction(
            action_size=action_size,
            observation_components=self.observation_components,
            state_encoding_method=self.state_encoding_method,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            **self.state_encoding_config
        )

    def get_params(self, deep=False):
        return dict(
            state_encoding_method=self.state_encoding_method,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            **self.state_encoding_config
        )


def resolve_observation_encoder(name, **config):
    if name == 'drr':
        from recsys_mdp.models.state_representation import StateReprModuleWithAttention
        return StateReprModuleWithAttention(
            n_users, n_items, id_embedding_dim, memory_size, learn,
            attention_hidden_size,
            initial_user_embeddings=initial_user_embeddings,
            initial_item_embeddings=initial_item_embeddings
        )
    elif name == 'adrr':
        from recsys_mdp.models.state_representation import StateReprModule
        return StateReprModule(
            n_users, n_items, id_embedding_dim,
            memory_size, learn,
            initial_user_embeddings=initial_user_embeddings,
            initial_item_embeddings=initial_item_embeddings
        )
    elif name == 'concat':
        from recsys_mdp.models.state_representation import ConcatState
        return ConcatState()
