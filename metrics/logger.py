import torch
class Logger():
    def __init__(self, interactive_mdp, user_interests, fake_mdp,
                 top_k, static_scorers, interactive_scorers):
        #self.model = model
        self.discrete = False
        self.interactive_mdp = interactive_mdp # d3rlpy mdp
        self.fake_mdp = fake_mdp # mdp for static metrics calculation, builded based on train
        self.user_interests = user_interests
        self.static_scorers = static_scorers
        self.interactive_scorers = interactive_scorers
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
        for scorer in self.static_scorers:
            scorer(pred_user_interests, self.user_interests, self.top_k)


    def interactive_log(self, model):
        interaction_result = []
        for episode in self.interactive_mdp:
            user = int(episode.observations[0][-1])
            print(self.user_interests.keys())
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

        for iscorer in self.interactive_scorers:
            iscorer(interaction_result)

    def visual_log(self):
        pass
