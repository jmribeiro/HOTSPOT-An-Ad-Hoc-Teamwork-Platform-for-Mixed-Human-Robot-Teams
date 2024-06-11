import numpy as np

from yaaf import normalize, uniform_distribution
from yaaf.agents import Agent
from yaaf.environments.markov import MarkovDecisionProcess
from yaaf.policies import greedy_policy

# ########## #
# Heuristics #
# ########## #

class TEQMDP(Agent):

    def __init__(self, pomdp):

        super(TEQMDP, self).__init__("Transition Entropy QMDP Agent")

        teqmdp = self.teqmdp = self.teqmdp(pomdp)
        self.pomdp = pomdp

        self.q_values = pomdp.q_values
        self._policy = pomdp.policy

        self.information_q_values = teqmdp.q_values
        self.information_policy = teqmdp.policy

        self.num_actions = pomdp.num_actions

        self.reset()

    def reset(self):
        self.belief = self.pomdp.miu
        self._policy = self.compute_next_policy()

    def compute_next_policy(self):
        Q = self.q_values
        Q_info = self.information_q_values
        q_values = self.transition_entropy_q_values(self.belief, Q, Q_info)
        return greedy_policy(q_values)

    def policy(self, observation=None):
        return self._policy

    def _reinforce(self, timestep):
        a = timestep.action
        next_obs = timestep.next_observation
        self.belief = self.pomdp.belief_update(self.belief, a, next_obs)
        self._policy = self.compute_next_policy()

    @staticmethod
    def teqmdp(pomdp):
        return MarkovDecisionProcess(
            name=pomdp.spec.id.replace("pomdp", "teqmdp"),
            states=pomdp.states,
            actions=pomdp.actions,
            transition_probabilities=pomdp.transition_probabilities,
            rewards=TEQMDP.information_reward(pomdp),
            discount_factor=pomdp.discount_factor,
            initial_state_distribution=pomdp.initial_state_distribution,
            state_meanings=pomdp.state_meanings,
            action_meanings=pomdp.action_meanings
        )

    @staticmethod
    def information_gain(pomdp):
        dH = np.zeros((pomdp.num_actions, pomdp.num_observations))
        for a in range(pomdp.num_actions):
            Pa_sum = np.sum(pomdp.P[a], axis=0)
            mu = np.diag(Pa_sum).dot(pomdp.O[a]).T
            msum = np.sum(mu, axis=1)
            for z in range(pomdp.num_observations):
                if msum[z] > 0:
                    aux = 0
                    for x in range(pomdp.num_states):
                        mu[z, x] = mu[z, x] / msum[z]
                        if mu[z, x] > 0:
                            aux += mu[z, x] * np.log(mu[z, x]) / np.log(pomdp.num_states)
                    dH[a, z] = 1 + aux
                else:
                    dH[a, z] = 0
        return dH

    @staticmethod
    def reward_gain(pomdp):
        d_r = np.zeros((pomdp.num_actions, pomdp.num_observations))
        for a in range(pomdp.num_actions):
            Pa_sum = np.sum(pomdp.P[a], axis=0) / pomdp.num_states
            for z in range(pomdp.num_observations):
                d_r[a, z] = np.max((Pa_sum * pomdp.O[a, :, z])[None, :].dot(pomdp.R), axis=1)
        return d_r

    @staticmethod
    def information_reward(pomdp):
        rG = np.zeros((pomdp.num_states, pomdp.num_actions))
        G = TEQMDP.information_gain(pomdp) # * TEQMDP.reward_gain(pomdp)
        for a in range(pomdp.num_actions):
            rG[:, a] = pomdp.P[a].dot(pomdp.O[a]).dot(G[a, :, None])[:, 0]
        return rG

    @staticmethod
    def normalized_entropy(belief):
        H = 0
        num_states = len(belief)
        for x in range(num_states):
            if belief[x] > 0:
                H -= belief[x] * np.log(belief[x]) / np.log(num_states)
        return H

    @staticmethod
    def transition_entropy_q_values(belief, q_values, information_q_values):
        H = TEQMDP.normalized_entropy(belief)
        TEQ = belief.dot((1 - H) * q_values + H * information_q_values)
        return TEQ

# ############# #
# Ad Hoc Agents #
# ############# #

class TEBOPA(Agent):

    def __init__(self, pomdps, informative_actions):

        super(TEBOPA, self).__init__("Transition Entropy BOPA")

        teqmdps = self.teqmdps = [TEQMDP.teqmdp(pomdp) for pomdp in pomdps]

        self.pomdps = pomdps
        self.num_pomdps = len(self.pomdps)

        self.q_values = [pomdp.q_values for pomdp in pomdps]
        self.policies = [pomdp.policy for pomdp in pomdps]

        self.information_q_values = [teqmdp.q_values for teqmdp in teqmdps]
        self.information_policies = [teqmdp.policy for teqmdp in teqmdps]

        self.num_actions = pomdps[0].num_actions
        self.informative_actions = informative_actions

        self.reset()

    def reset(self):
        self.pomdp_probabilities = uniform_distribution(self.num_pomdps)
        self.beliefs = [pomdp.miu for pomdp in self.pomdps]
        self._policy = self.compute_next_policy()

    def compute_model_policy(self, k, combined_belief):
        Q = self.q_values[k]
        Q_info = self.information_q_values[k]
        belief = self.beliefs[k]
        q_values = self.combined_transition_entropy_q_values(belief, combined_belief, Q, Q_info)
        return q_values

    @staticmethod
    def combined_transition_entropy_q_values(pomdp_belief, combined_belief, q_values, information_q_values):
        H = TEQMDP.normalized_entropy(combined_belief)
        q_values_weight = (1 - H)
        localization_weight = H
        TEQ = pomdp_belief.dot((q_values_weight * q_values) + (localization_weight * information_q_values))
        return TEQ

    def compute_next_policy(self):
        q_values = np.zeros(self.num_actions)
        combined_belief = self.pomdp_probabilities.dot(np.array(self.beliefs))
        combined_entropy = TEQMDP.normalized_entropy(combined_belief)
        for k in range(self.num_pomdps):
            model_q_values = self.compute_model_policy(k, combined_belief)
            q_values += model_q_values * self.pomdp_probabilities[k]
        self.combined_entropy = combined_entropy
        self.combined_belief = combined_belief
        return greedy_policy(q_values)

    def policy(self, observation):
        return self._policy

    def _reinforce(self, timestep):
        a = timestep.action
        next_obs = timestep.next_observation
        #if a not in self.informative_actions:
        self.update_beliefs_over_pomdps(a, next_obs)
        self.update_beliefs_over_states(a, next_obs)
        self._policy = self.compute_next_policy()
        timestep.info["beliefs over pomdps"] = self.pomdp_probabilities
        return timestep.info

    def update_beliefs_over_pomdps(self, a, next_obs):
        accumulators = np.zeros(self.num_pomdps)
        for k, pomdp in enumerate(self.pomdps):
            belief = self.beliefs[k]
            z = pomdp.observation_index(next_obs)
            accumulator = 0
            for x in range(pomdp.num_states):
                dot = pomdp.O[a, :, z].dot(pomdp.P[a, x])
                accumulator += dot * belief[x]
            accumulator *= self._policy[a] * self.pomdp_probabilities[k]
            accumulators[k] = accumulator
        self.pomdp_probabilities = normalize(accumulators)

    def update_beliefs_over_states(self, a, next_obs):
        for k, pomdp in enumerate(self.pomdps):
            states = pomdp.states
            next_belief = pomdp.belief_update(self.beliefs[k], a, next_obs)
            self.beliefs[k] = next_belief
            #print(f"MLS {k}: {states[next_belief.argmax()]}")

class QMDPAgent(Agent):

    def __init__(self, pomdp):
        super(QMDPAgent, self).__init__("QMDP")
        self._pomdp = pomdp
        self._q_values = pomdp.q_values
        self.reset()

    def reset(self):
        self.belief = self._pomdp.miu

    def policy(self, observation=None):
        q_values = self.belief.dot(self._q_values)
        return greedy_policy(q_values)

    def _reinforce(self, timestep):
        _, action, _, next_obs, _, _ = timestep
        self.belief = self._pomdp.belief_update(self.belief, action, next_obs)

class MLSAgent(Agent):

    def __init__(self, pomdp):
        super(MLSAgent, self).__init__("MLS")
        self._pomdp = pomdp
        self._q_values = pomdp.q_values
        self.reset()

    def reset(self):
        self.belief = self._pomdp.miu

    def policy(self, observation=None):
        most_likely_state = self.belief.argmax()
        q_values = self._q_values[most_likely_state]
        return greedy_policy(q_values)

    def _reinforce(self, timestep):
        _, action, _, next_obs, _, _ = timestep
        self.belief = self._pomdp.belief_update(self.belief, action, next_obs)
