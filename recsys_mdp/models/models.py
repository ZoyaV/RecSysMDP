from typing import Sequence, ClassVar

import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch import Encoder, EncoderWithAction

from recsys_mdp.mdp.utils import isnone
from recsys_mdp.models.state_representation import (
    StateReprModuleWithAttention, StateReprModule, FullHistory
)


DEFAULT_STATE_KEYS = ['user', 'item', 'score']


class ActorEncoder(nn.Module, Encoder):
    def __init__(
            self, user_num, item_num, embedding_dim,
            hidden_dim, memory_size, feature_size,
            state_repr_name ="drr",
            state_keys = None,
            freeze_emb = False,
            attention_hidden_size = 0,
            initial_user_embeddings=None,
            initial_item_embeddings=None
    ):
        super().__init__()

        self.state_repr_name = state_repr_name
        self.state_keys = isnone(state_keys, DEFAULT_STATE_KEYS)
        if state_repr_name == 'drr':
            self.state_repr = StateReprModuleWithAttention(
                user_num, item_num, embedding_dim, memory_size, freeze_emb,
                attention_hidden_size,
                initial_user_embeddings=initial_user_embeddings,
                initial_item_embeddings=initial_item_embeddings
            )
        elif state_repr_name == 'adrr':
            self.state_repr = StateReprModule(
                user_num, item_num, embedding_dim,
                memory_size,freeze_emb,
                initial_user_embeddings=initial_user_embeddings,
                initial_item_embeddings=initial_item_embeddings
            )
        elif state_repr_name == 'full_history':
            self.state_repr = FullHistory(
                user_num, item_num, embedding_dim,
                memory_size, freeze_emb,
                state_keys=state_keys,
                initial_user_embeddings=initial_user_embeddings,
                initial_item_embeddings=initial_item_embeddings,
            )
        self.feature_size = feature_size
        self.memory_size = memory_size

        state_repr_size = embedding_dim * self.state_repr.out_embeddings
        if 'score' in self.state_keys:
            state_repr_size += self.memory_size
        self.layers = nn.Sequential(
            nn.Linear( state_repr_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_size),
        )

        self.initialize()

    def initialize(self):
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        """
        :param user: user batch
        :param memory: memory batch
        :return: output, vector of the size `feature_size`        """

        user = x[:, -1]
        memory = x[:, :self.memory_size]
        score = x[:, self.memory_size:-1]

        # try:
        state = self.state_repr(user, memory, score)
        # except Exception as e:
        #     print(e)
        #     print("USER: ", user)
        #     print("MEMORY: ", memory)
        #     print("SCORES: ", score)
        #     print(x)
        #     exit()
        return torch.relu(self.layers(state))

    def get_feature_size(self):
        return self.feature_size


class ActorEncoderWithAction(nn.Module, EncoderWithAction):
    def __init__(
            self, action_size, user_num, item_num,
            embedding_dim, hidden_dim, memory_size, feature_size,
            state_repr_name="drr", state_keys=None,
            freeze_emb=False, attention_hidden_size=0,
            initial_user_embeddings=None, initial_item_embeddings=None
    ):
        super().__init__()
        self.state_encoder = ActorEncoder(
            user_num=user_num, item_num=item_num,
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            memory_size=memory_size, feature_size=feature_size,
            state_repr_name=state_repr_name, state_keys=state_keys,
            freeze_emb=freeze_emb, attention_hidden_size=attention_hidden_size,
            initial_user_embeddings=initial_user_embeddings,
            initial_item_embeddings=initial_item_embeddings,
        )

        self.feature_size = feature_size
        self.fc1 = nn.Linear(feature_size + action_size, feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)

    def forward(self, x, action): # action is also given
        state = self.state_encoder(x)
        h = torch.cat([state, action], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h

    def get_feature_size(self):
        return self.feature_size


class ActorEncoderFactory(EncoderFactory):
    TYPE: ClassVar[str] = "custom"

    def __init__(
            self,
            user_num, item_num,
            embedding_dim, hidden_dim,
            memory_size,
            feature_size,
            state_repr_name = 'drr',
            state_keys=None,
            freeze_emb = False,
            attention_hidden_size = 0,
            initial_user_embeddings=None,
            initial_item_embeddings=None
    ):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.feature_size = feature_size
        self.state_repr_name = state_repr_name
        self.state_keys = state_keys
        self.freeze_emb = freeze_emb
        self.attention_hidden_size = attention_hidden_size
        self.initial_user_embeddings = initial_user_embeddings
        self.initial_item_embeddings = initial_item_embeddings

    def create(self, observation_shape: Sequence[int]) -> Encoder:
        return ActorEncoder(
            user_num=self.user_num,
            item_num=self.item_num,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            memory_size=self.memory_size,
            feature_size=self.feature_size,
            state_repr_name=self.state_repr_name,
            state_keys=self.state_keys,
            freeze_emb=self.freeze_emb,
            attention_hidden_size=self.attention_hidden_size,
            initial_user_embeddings=self.initial_user_embeddings,
            initial_item_embeddings=self.initial_item_embeddings
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        return ActorEncoderWithAction(
            action_size=action_size,
            user_num=self.user_num,
            item_num=self.item_num,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            memory_size=self.memory_size,
            feature_size=self.feature_size,
            state_repr_name=self.state_repr_name,
            state_keys=self.state_keys,
            freeze_emb=self.freeze_emb,
            attention_hidden_size=self.attention_hidden_size,
            initial_user_embeddings=self.initial_user_embeddings,
            initial_item_embeddings=self.initial_item_embeddings
        )

    def get_params(self, deep=False):
        return {
            'user_num': self.user_num,
            'item_num': self.item_num,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'memory_size': self.memory_size,
            'feature_size': self.feature_size,
            'state_repr_name':self.state_repr_name,
            'attention_hidden_size': self.attention_hidden_size,
            'initial_user_embeddings': self.initial_user_embeddings,
            'initial_item_embeddings': self.initial_item_embeddings
        }
