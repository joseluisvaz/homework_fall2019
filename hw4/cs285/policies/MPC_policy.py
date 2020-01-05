import numpy as np
import numpy.matlib

from .base_policy import BasePolicy
import cs285.infrastructure.utils as utils


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        seqs = np.zeros((num_sequences, horizon, self.ac_dim))
        for i in range(num_sequences):
            seqs[i] = np.random.uniform(self.ac_space.low, self.ac_space.high, (horizon, self.ac_dim))
        return seqs

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        for model in self.dyn_models:

            # obtain ground truth states from the env
            pred_states_sequences = np.zeros((self.N, self.horizon, self.ob_dim))

            # predict states using the model and given action sequence and initial state
            ob = np.expand_dims(obs, 0)
            ob = numpy.matlib.repmat(ob, self.N, 1)
            for j in range(self.horizon):
                pred_states_sequences[:, j, :] = ob
                ob = model.get_prediction(ob, candidate_action_sequences[:, j], self.data_statistics)

            rew_seqs = []
            for i in range(self.N):
                pred_states_sequence = pred_states_sequences[i]
                action_sequence = candidate_action_sequences[i]
                rew_seqs.append(sum(np.squeeze(self.env.get_reward(pred_states_sequence, action_sequence))))

            predicted_rewards_per_ens.append(rew_seqs)

        predicted_rewards_per_ens = np.squeeze(predicted_rewards_per_ens)
        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        predicted_rewards = np.mean(predicted_rewards_per_ens, axis=0)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards)
        best_action_sequence = candidate_action_sequences[best_index]
        action_to_take = best_action_sequence[0]
        return action_to_take[None] # the None is for matching expected dimensions
