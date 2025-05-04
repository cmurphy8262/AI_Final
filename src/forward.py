import numpy as np

def forward(self, observations):
  # Initialize the forward probabilities
  alpha = np.zeros((len(self.hidden_states), len(observations)))

  # Initialize the first column of alpha using initial probabilities and emission probabilities
  for s in range(len(self.hidden_states)):
      alpha[s, 0] = self.initial_probs[s] * self.emission_probs[s, observations[0]]

  # Iterate through the remaining observations
  for t in range(1, len(observations)):
      for s in range(len(self.hidden_states)):
          for prev_s in range(len(self.hidden_states)):
              alpha[s, t] += alpha[prev_s, t-1] * self.transition_probs[prev_s, s]
          alpha[s, t] *= self.emission_probs[s, observations[t]]

  return alpha