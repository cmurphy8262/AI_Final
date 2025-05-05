# Hidden Markov Model for Speech Recognition
By Dominic Cork and Connor Murphy

## Overview
This project implements a simple Automatic Speech Recognition (known as an ASR) system using Hidden Markov Models (HMMs). The system is trained and tested using the Google Speech Commands dataset, and it has the ability to recognize spoken keywords based on extracted audio features.

## Features
- Implements core HMM components: forward algorithm, Viterbi decoding, and model fitting

- Uses MFCCs (Mel-Frequency Cepstral Coefficients) for feature extraction

- Applies KMeans clustering to discretize audio into observation sequences

- Evaluates model accuracy on real-world spoken word data

- Supports configurable keyword selection and dataset sizes

## Setup
1. Clone the Repository
  - git clone https://github.com/cmurphy8262/AI_Final.git
  - cd AI_Final
2. Install Dependencies
  - pip install -r requirements.txt
3. Run the Project
  - python src/main.py
