def evaluate_test_sample(hmm, audio_path):

    # Clustered observation sequence from real audio
    obs_sequence = hmm.process_audio(audio_path)

    # Run Forward Algorithm
    forward_probs = hmm.forward(obs_sequence)
    print("Forward probabilities:")
    print(forward_probs)

    # Run Viterbi Algorithm
    viterbi_path, viterbi_prob = hmm.viterbi(obs_sequence)
    print("\nViterbi path:")
    print(viterbi_path)
    print("\nViterbi probability:")
    print(viterbi_prob)

    return viterbi_path, viterbi_prob, forward_probs