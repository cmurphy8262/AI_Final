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