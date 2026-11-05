import pickle
import numpy as np

with open("tracks.pkl", "rb") as f:
    data = pickle.load(f)

trajectories = data["trajectories"]
lengths = [len(t) for t in trajectories.values()]

print("Número de tracks:", len(lengths))
print("Mínimo / máximo:", min(lengths), max(lengths))
print("Percentis 10/50/90:", np.percentile(lengths, [10, 50, 90]))
print("Tracks com menos de 15 frames:", sum(l < 15 for l in lengths))