import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from recsys_mdp.models.state_representation import StateReprModuleWithAttention, StateReprModule, FullHistory


class ActorEncoder(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim,
                 hidden_dim, memory_size, feature_size,
                 state_repr_name ="drr",
                 freeze_emb = False,
                 attention_hidden_size = 0,
                 use_als = False,
                 user_emb = None,
                 item_emb = None):
        super().__init__()

        self.state_repr_name = state_repr_name
        if state_repr_name == 'drr':
            self.state_repr = StateReprModuleWithAttention(
                user_num, item_num, embedding_dim, memory_size, freeze_emb,
                attention_hidden_size, use_als, user_emb, item_emb
            )
        elif state_repr_name == 'adrr':
            self.state_repr = StateReprModule(
                user_num, item_num, embedding_dim,
                memory_size,freeze_emb, use_als, user_emb, item_emb
            )
        elif state_repr_name == 'full_history':
            self.state_repr = FullHistory(
                user_num, item_num, embedding_dim,
                memory_size, freeze_emb, use_als, user_emb, item_emb
            )
        self.feature_size = feature_size
        self.memory_size = memory_size
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * self.state_repr.out_embeddings, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_size),
        )

        self.initialize()

    def initialize(self):
        """weight init"""
       # return
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        """
        :param user: user batch
        :param memory: memory batch
        :return: output, vector of the size `feature_size`
        """

        # TODO: how to check is user in x or score in x?
        user = x[:,-1]
        memory = x[:, :self.memory_size]
        score = x[:, self.memory_size:-1]
      #  print(user)
       # exit()
        try:
            state = self.state_repr(user, memory, score)
        except Exception as e:
            print(e)
            print(user)
            print(memory)
            print(score)
            print(x)
            exit()
        return torch.relu(self.layers(state))

    def get_feature_size(self):
        return self.feature_size

class ActorEncoderWithAction(nn.Module):
    def __init__(self,action_size, user_num, item_num, embedding_dim,
                 hidden_dim, memory_size, feature_size,
                 state_repr_name ="drr",
                 freeze_emb = False,
                 attention_hidden_size = 0,
                 use_als = False,
                 user_emb = None,
                 item_emb = None):

        self.state_encoder = ActorEncoder(
            user_num,
            item_num,
            embedding_dim,
            hidden_dim,
            memory_size,
            feature_size,
            state_repr_name,
            freeze_emb,
            attention_hidden_size,
            use_als,
            user_emb,
            item_emb,
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
    TYPE = 'custom'

    def __init__(
            self,
            user_num,
            item_num,
            embedding_dim,
            hidden_dim,
            memory_size,
            feature_size,
            state_repr_name = 'drr',
            freeze_emb = False,
            attention_hidden_size = 0,
            use_als = False,
            user_emb = None,
            item_emb = None


    ):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.feature_size = feature_size
        self.state_repr_name = state_repr_name
        self.freeze_emb = freeze_emb
        self.attention_hidden_size = attention_hidden_size
        self.use_als = use_als
        self.user_emb = user_emb
        self.item_emb = item_emb

    def create(self, observation_shape, action_size=None, discrete_action=False):
            if action_size is None:
                return ActorEncoder(
                    self.user_num,
                    self.item_num,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.memory_size,
                    self.feature_size,
                    self.state_repr_name,
                    self.freeze_emb,
                    self.attention_hidden_size,
                    self.use_als,
                    self.user_emb,
                    self.item_emb
                )
            else:
                ActorEncoderWithAction(action_size, self.user_num,
                    self.item_num,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.memory_size,
                    self.feature_size,
                    self.state_repr_name,
                    self.freeze_emb,
                    self.attention_hidden_size,
                    self.use_als,
                    self.user_emb,
                    self.item_emb
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
            'attention_hidden_size': self.attention_hidden_size
        }



def load_embeddings(path_to_model="model_final.pt", model_params=[943, 1682, 8, 16, 5, False, 64]):
    the_model = ActorEncoder(*model_params)
    the_model.load_state_dict(torch.load(path_to_model))
    user_emb = the_model.state_repr.user_embeddings.weight.data
    item_emb = the_model.state_repr.item_embeddings.weight.data
    return user_emb, item_emb