import random

def keyword_file_lists(dataset_dir, keywords, validation_percentage, testing_percentage, max_test):

  # Containers for training and testing files
  train_files = {kw: [] for kw in keywords}
  test_files = {kw: [] for kw in keywords}

  # Loop through each target keyword folder
  for keyword in keywords:
    keyword_path = os.path.join(dataset_dir, keyword)
    if not os.path.isdir(keyword_path):
      continue # Skip if the folder does not exist

    # Get all .wav files in the folder and shuffle for randomness
    all_files = [f for f in os.listdir(keyword_path) if f.endswith(".wav")]
    random.shuffle(all_files) # Shuffle to sample randomly

    # Loop through all files in the current keyword folder
    for filename in all_files:
      filepath = os.path.join(keyword_path, filename)
      set_type = which_set(filepath, validation_percentage, testing_percentage)

      # Assign the file to a list based on the partition result
      if set_type == 'training':
        train_files[keyword].append(filepath)
      elif set_type == 'testing':
        test_files[keyword].append(filepath)

      # Stop looping once training and testing limits are reached (optimize runtime)
      if len(test_files[keyword]) >= max_test:
        break


  return train_files, test_files