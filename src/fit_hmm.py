import numpy as np
from collections import defaultdict

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