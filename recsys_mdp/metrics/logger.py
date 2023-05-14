import numpy as np
import torch


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
    def __init__(
            self, interactive_mdp, user_interests, fake_mdp,
            top_k, static_scorers, interactive_scorers, visual_loggers,
            wandb_logger=None
    ):
        #self.model = model
        self.discrete = True
        self.interactive_mdp = interactive_mdp # d3rlpy mdp
        self.fake_mdp = fake_mdp # mdp for static metrics calculation, builded based on train
        self.user_interests = user_interests
        self.static_scorers = static_scorers
        self.interactive_scorers = interactive_scorers
        self.visual_loggers = visual_loggers
        self.top_k = top_k
        self.wandb_logger = wandb_logger

    def get_value(self, model, obs, item = None):
        if self.discrete:
            observations_cuda = torch.from_numpy(obs).cpu()
            with torch.no_grad():
                algo = model.__class__.__name__
               # print(algo)
                if "BC" in algo:
                    predicted_rat = model.impl._imitator(observations_cuda.to(torch.float32)).cpu().detach().numpy()
                else:
                    predicted_rat = model._impl._q_func(observations_cuda).cpu().detach().numpy()
                predicted_rat = (predicted_rat - predicted_rat.min()) / (predicted_rat.max() - predicted_rat.min())
        else:
            predicted_rat = model.predict(obs)
        if item is None:
            return predicted_rat
        else:
           # print(predicted_rat[0].shape)
            return predicted_rat[0][item]
    def static_log(self, model):
        if len(self.static_scorers) == 0:
            return {"none" : None}
        obs = self.fake_mdp
        predicted_rat = self.get_value(model, obs)

        users = obs[:,-1]
        #print(self.discrete)
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
            original_rewards = episode.rewards[:self.top_k]
            obs = episode.observations[0]
            not_interactive_items = model.predict(episode.observations)
            interactive_items = []
            predicted_values = []
            for i in range(self.top_k):
                new_item = model.predict([obs])[0]
                orig_obs = episode.observations[i:i+1]
                if i>=len(original_items):
                    break
                try:
                    value = self.get_value(model, orig_obs, original_items[i])
                except Exception as e:
                    print(e)
                    break

                predicted_values.append(value)
                interactive_items.append(new_item)

                new_obs = obs.copy()
                new_obs[:-3] = new_obs[1:-2]
                new_obs[-2] = new_item
                obs = new_obs.copy()


            interaction_result.append((interactive_items, original_items,
                                       self.user_interests[user], not_interactive_items,
                                       original_rewards, predicted_values))

        results = dict()
        for iscorer_key in self.interactive_scorers:
            results[iscorer_key] = self.interactive_scorers[iscorer_key](interaction_result)
        return results

    def visual_log(self, model, log_resuls = None):

        if len( self.visual_loggers) > 0:
            obs = self.interactive_mdp.observations
            predcted_items = model.predict(obs)
            unique_items = torch.from_numpy(np.arange(1,1000)).long()
            unique_users = torch.from_numpy(np.arange(1,1000)).long()

            users_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.user_embeddings(unique_users)
            items_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.item_embeddings(unique_items)
            users_emb = users_emb.detach().numpy()
            items_emb = items_emb.detach().numpy()
            discrete = True
            visual_info = {'total_prediction':predcted_items,
                           'vals': [users_emb,items_emb],
                           'names':['user_emb','items_emb'],
                           'discrete': discrete}

        for visual_logger in self.visual_loggers:
            visual_logger(**visual_info)

        flattened_dict = flatten_dict_keys(log_resuls)
        print(flattened_dict)
        if self.wandb_logger:
            self.wandb_logger.log(flattened_dict)
