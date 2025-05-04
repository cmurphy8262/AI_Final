import os
import re
import hashlib
import random

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):

  # Ignore anything after '_nohash_' in the file name when deciding which set to put a .wav in
  base_name = os.path.basename(filename)

  # Decide whether a file should go into the training, testing, or validation sets. Also want to keep existing files in the same set even if more files are added.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)

  # Hash the filename then use that to generate a probability value that can be used to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


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