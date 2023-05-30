import pickle

import yaml

from recsys_mdp.experiments.utils.algorithm_constuctor import init_algo, init_hidden_state_encoder
from recsys_mdp.experiments.utils.mdp_constructor import load_data, make_mdp
from recsys_mdp.utils.run.config import read_config
from recsys_mdp.mdp.utils import to_d3rlpy_form_ND


def load_checkpoint(model, model_name, step=-1):
    if step == -1:
        path = f'checkpoints/{model_name}/{model_name}.pt'
    else:
        path = f'checkpoints/{model_name}/{model_name}_{step}.pt'
    model.load_model(path)
