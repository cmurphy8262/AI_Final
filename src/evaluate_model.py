def evaluate_model(hmm, test_files, keywords):
  correct = 0 # number of correct predictions
  total = 0 # total number of test samples

  # Loop through each keyword in the test set
  for keyword in keywords:
    for file_path in test_files[keyword]:
      try:
        # Convert audio to observation sequence using MFCC & KMeans
        obs_sequence = hmm.process_audio(file_path)

        # Run Viterbi algorithm to get most likely hidden states sequence
        predicted_states, _ = hmm.viterbi(obs_sequence)

        # Predict the label based on the most common state in the Viterbi path
        # predicted_label = assign_most_common_label(predicted_states, hmm.hidden_states)
        # Predict the label based on the last hidden state
        predicted_label = hmm.hidden_states[predicted_states[-1]]


        # Compare prediction with true label
        if predicted_label == keyword:
          correct += 1
        total += 1
      except Exception as e:
        print(f"Error processing {file_path}: {e}")

  # Calculate accuracy
  accuracy = correct / total if total > 0 else 0
  print(f"\nModel accuracy on test set: {accuracy * 100:.2f}% ({correct}/{total})")