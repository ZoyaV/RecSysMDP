import torch
import numpy as np

import wandb

def flatten_dict_keys(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_keys(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
class Logger():
    def __init__(self, interactive_mdp, user_interests, fake_mdp,
                 top_k, static_scorers, interactive_scorers, visual_loggers):
        #self.model = model
        self.discrete = False
        self.interactive_mdp = interactive_mdp # d3rlpy mdp
        self.fake_mdp = fake_mdp # mdp for static metrics calculation, builded based on train
        self.user_interests = user_interests
        self.static_scorers = static_scorers
        self.interactive_scorers = interactive_scorers
        self.visual_loggers = visual_loggers
        self.top_k = top_k

    def static_log(self, model):
        obs = self.fake_mdp
        if self.discrete:
            observations_cuda = torch.from_numpy(obs).cpu()
        if self.discrete:
            with torch.no_grad():
                predicted_rat = (model.get_values(observations_cuda)).cpu().detach().numpy()
                predicted_rat = (predicted_rat - predicted_rat.min()) / (predicted_rat.max() - predicted_rat.min())
        else:
            predicted_rat = model.predict(obs)

        users = obs[:,-1]
        pred_user_interests = [users, predicted_rat]

        results = dict()
        for scorer_key in self.static_scorers:
            results[scorer_key] = self.static_scorers[scorer_key](pred_user_interests, self.user_interests, self.top_k)
        return results


    def interactive_log(self, model):
        interaction_result = []
        for episode in self.interactive_mdp:
            user = int(episode.observations[0][-1])
            original_items = episode.actions
            obs = episode.observations[0]
            not_interactive_items = model.predict(episode.observations)
            interactive_items = []
            for _ in range(self.top_k):
                new_item = model.predict([obs])[0]
                interactive_items.append(new_item)

                new_obs = obs.copy()
                new_obs[:-3] = new_obs[1:-2]
                new_obs[-2] = new_item
                obs = new_obs.copy()
            interaction_result.append((interactive_items, original_items, self.user_interests[user], not_interactive_items))

        results = dict()
        for iscorer_key in self.interactive_scorers:
            results[iscorer_key] = self.interactive_scorers[iscorer_key](interaction_result)
        return results

    def visual_log(self, model, log_resuls = None):
        obs = self.interactive_mdp.observations
        predcted_items = model.predict(obs)

        unique_items = torch.from_numpy(np.arange(1,1000)).long()
        unique_users = torch.from_numpy(np.arange(1,1000)).long()
     #   print(len(obs[:,-1].ravel()))
        users_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.user_embeddings(unique_users)
        items_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.item_embeddings(unique_items)
        users_emb = users_emb.detach().numpy()
        items_emb = items_emb.detach().numpy()
        discrete = True
        visual_info = {'total_prediction':predcted_items,
                       'vals': [users_emb,items_emb],
                       'names':['user_emb','items_emb'],
                       'discrete': discrete}

     #   print(self.visual_loggers)
        for visual_logger in self.visual_loggers:
            visual_logger(**visual_info)

        flattened_dict = flatten_dict_keys(log_resuls)
        wandb.log(flattened_dict)
        pass
