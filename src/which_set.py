import os
import re
import hashlib

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