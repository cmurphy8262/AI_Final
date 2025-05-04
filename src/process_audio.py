import librosa
from sklearn.cluster import KMeans

def process_audio(self, audio_path):
  # Load audio file
  y, sr = librosa.load(audio_path, sr=None)

  # Extract MFCCs
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
  mfcc = mfcc.T

  # Cluster MFCCs
  kmeans = KMeans(n_clusters=self.n_cluster)
  kmeans.fit(mfcc)
  clustered_observations = kmeans.label_.tolist()

  return clustered_observations