import numpy as np
import librosa
from sklearn.cluster import KMeans

class HMM:
    def __init__(self, states, observations, n_mfcc, n_clusters):
        self.hidden_states = states # the hidden states within the model
        self.observations = observations # the number of observations present in the model
        self.num_hidden_states = len(states) # total number of hidden states
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters


    def fit_hmm(hmm, train_files,  keywords):

      # Initialize count matrices
      num_states = len(hmm.hidden_states)
      num_obs = hmm.n_clusters
      transition_counts = np.zeros((num_states, num_states))
      emission_counts = np.zeros((num_states, num_obs))
      initial_counts = np.zeros(num_states)

      # Map each keyword to a unique hidden state index
      state_mapping = {kw: i for i, kw in enumerate(keywords)}



      for keyword in keywords:
        state_index = state_mapping[keyword]
        for file_path in train_files[keyword]:
          try:

            # Get observation sequence for this audio
            obs_sequence = hmm.process_audio(file_path)

            # Initial state count
            initial_counts[state_index] += 1

            # Emission counts
            for obs in obs_sequence:
              emission_counts[state_index][obs] += 1

            # Transition counts
            for i in range(len(obs_sequence) - 1):
              transition_counts[state_index][state_index] += 1

          except Exception as e:
            print(f"Error processing {file_path}: {e}")

      # Normalize
      hmm.initial_probs = initial_counts / np.sum(initial_counts)
      hmm.transition_probs = transition_counts / np.maximum(np.sum(transition_counts, axis=1, keepdims=True), 1)
      hmm.emission_probs = emission_counts / np.maximum(np.sum(emission_counts, axis=1, keepdims=True), 1)

      return hmm

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

    def process_audio(self, audio_path):
      # Load audio file
      y, sr = librosa.load(audio_path, sr=None)

      # Extract MFCCs
      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
      mfcc = mfcc.T

      # Cluster MFCCs
      kmeans = KMeans(n_clusters=self.n_clusters)
      kmeans.fit(mfcc)
      clustered_observations = kmeans.labels_.tolist()

      return clustered_observations

    def viterbi(self, observations):
      # Initialize the Viterbi probabilities and backpointers
      viterbi_probs = np.zeros((len(self.hidden_states), len(observations)))
      backpointers = np.zeros((len(self.hidden_states), len(observations)), dtype=int)

      # Initialize the first column of Viterbi probabilities
      for s in range(len(self.hidden_states)):
          viterbi_probs[s, 0] = self.initial_probs[s] * self.emission_probs[s, observations[0]]

      # Iterate through the remaining observations
      for t in range(1, len(observations)):
          for s in range(len(self.hidden_states)):
              max_prob = 0
              max_prev_state = 0
              for prev_s in range(len(self.hidden_states)):
                  prob = viterbi_probs[prev_s, t-1] * self.transition_probs[prev_s, s]
                  if prob > max_prob:
                      max_prob = prob
                      max_prev_state = prev_s
              viterbi_probs[s, t] = max_prob * self.emission_probs[s, observations[t]]
              backpointers[s, t] = max_prev_state

      # Find the most likely sequence of hidden states
      best_path_prob = np.max(viterbi_probs[:, -1])
      best_path_end_state = np.argmax(viterbi_probs[:, -1])
      best_path = [best_path_end_state]

      for t in range(len(observations) - 2, -1, -1):
          best_path.insert(0, backpointers[best_path[0], t + 1])

      return best_path, best_path_prob
