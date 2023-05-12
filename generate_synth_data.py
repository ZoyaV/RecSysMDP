import os

import numpy as np
import pandas as pd
from pathlib import Path
from d3rlpy.base import LearnableBase

from recsys_mdp.generators.scenarios.mdp_next_item_integration import NextItemEnvironment, TypesResolver
from recsys_mdp.generators.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
import yaml

def run(config_path = "recsys_mdp/generators/configs/mdp_next_item_integration.yaml"):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    path_object = Path(config_path)
    config_class = GlobalConfig(
                config=config, config_path=path_object, type_resolver=TypesResolver()
            )

    env_conf = config['env']
    env: NextItemEnvironment = config_class.resolve_object(
        env_conf, object_type_or_factory=NextItemEnvironment
    )

    model_conf = config['model']
    model: LearnableBase = config_class.resolve_object(
                model_conf | dict(use_gpu=False),
                n_actions=env.n_items
            )
    return generate_episode(env, model)
def generate_episode(env, model):
    env, model = env, model
    user_id = env.reset()
    trajectory = []

    # [10 last item_ids] + [user_id]
    fake_obs = np.random.randint(0, 3521, 10).tolist() + [user_id]
    obs = np.asarray(fake_obs)

    while True:
        try:
            item_id = model.predict(obs.reshape(1, -1))[0]
        except:
            item_id = model.predict(obs[:10].reshape(1, -1))[0]
        obs[:9] = obs[1:10]
        obs[-2] = item_id

        timestamp = env.timestamp

        relevance, terminated = env.step(item_id)
        continuous_relevance, discrete_relevance = relevance
        trajectory.append((
            timestamp,
            user_id, item_id,
            continuous_relevance, discrete_relevance,
            terminated
        ))
        if terminated:
            break
    return trajectory

if __name__ == "__main__":
    trajectories = run()
    # Define the column names
    column_names = ['timestamp', 'user_idx', 'item_idx', 'relevance_cont', 'relevance_int', 'terminated']

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(trajectories, columns=column_names)

    # Create the "cases" directory if it does not exist
    directory = 'row_data/cases'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file name
    case_name = 'case_name'
    file_name = f"{directory}/{case_name}.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)