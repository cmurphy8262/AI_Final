# Hidden Markov Model for Speech Recognition
By Dominic Cork and Connor Murphy

## Overview
This project implements a simple Automatic Speech Recognition (known as an ASR) system using Hidden Markov Models (HMMs). The system is trained and tested using the Google Speech Commands dataset, and it has the ability to recognize spoken keywords based on extracted audio features.


## Setup
1. Clone the Repository
  - git clone https://github.com/cmurphy8262/AI_Final.git
  - cd AI_Final
2. Install Dependencies
  - pip install -r requirements.txt
3. Run the Project
  - python src/main.py


## Features
- Implements core HMM components: forward algorithm, Viterbi decoding, and model fitting

- Uses MFCCs (Mel-Frequency Cepstral Coefficients) for feature extraction

- Applies K-Means clustering to discretize audio into observation sequences

- Evaluates model accuracy on spoken word data

- Supports configurable keyword selection and dataset sizes


## File Descriptions
- main.py: Entry point for the project. Trains the HMM, evaluates one sample, and prints model accuracy.
- HMM.py: Defines the HMM class, which includes the forward and Viterbi algorithms, as well as audio processing.
- training.py: Contains the fit_hmm() function, which is used to estimate the transition, emission, and initial probabilities of the training data.
- evaluation.py: Includes evaluate_model() for computing accuracy, and evaluate_test_sample() to test and print model behavior on one sample.
- data_utils: Provides helper functions for partitioning the dataset and listing keyword audio files using deterministic hashing.

## Dependencies
- numpy
- librosa
- scikit-learn
