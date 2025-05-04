def main():
    # Define dataset path and target keywords
    dataset_dir = "/content/speech_commands"
    target_keywords = ["stop", "go"]

    # Build the file lists using deterministic hashing
    train_files, test_files = keyword_file_lists(dataset_dir=dataset_dir, keywords=target_keywords, validation_percentage=10, testing_percentage=10, max_test=100)

    # Create HMM object, define states and observations
    hmm = HMM(states=target_keywords, observations=list(range(6)), n_mfcc=13, n_clusters=6)

    # Fit the HMM to the training data
    fit_hmm(hmm, train_files, target_keywords)

    # Evaluate HMM functionality on a sample test file
    selected_keyword = "go"
    if not test_files[selected_keyword]:
      print(f"No test files found for '{selected_keyword}'")
      return

    audio_path = test_files[selected_keyword][0] # First test file in selected_keyword folder
    evaluate_test_sample(hmm, audio_path)

    # Evaluate the model accuracy
    evaluate_model(hmm, test_files, target_keywords)

if __name__ == "__main__":
    main()
