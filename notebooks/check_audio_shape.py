import numpy as np

X = np.load("../data/processed/audio/X.npy")
y = np.load("../data/processed/audio/y.npy")

print(X.shape)
print(y.shape)