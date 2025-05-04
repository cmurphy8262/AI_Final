import numpy as np

class HMM:
    def __init__(self, states, observations, n_mfcc, n_clusters):
        self.hidden_states = states # the hidden states within the model
        self.observations = observations # the number of observations present in the model
        self.num_hidden_states = len(states) # total number of hidden states
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters

        # generate random transition probabilities, then normalize
        self.transition_probs = np.random.rand(self.num_hidden_states, self.num_hidden_states)
        self.transition_probs = self.transition_probs / np.sum(self.transition_probs, axis=1, keepdims=True)

        # generate random emission probabilities, then normalize
        self.emission_probs = np.random.rand(len(self.hidden_states), len(self.observations))
        self.emission_probs = self.emission_probs / np.sum(self.emission_probs, axis=1, keepdims=True)  # Normalizing data

        # generate random initial probabilities, then normalize
        self.initial_probs = np.random.rand(len(self.hidden_states))
        self.initial_probs = self.initial_probs / np.sum(self.initial_probs) # Normalizing data