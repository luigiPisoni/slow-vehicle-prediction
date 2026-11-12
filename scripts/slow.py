import json
import os
import pickle
from collections import Counter

import numpy as np

# velocidade = deslocamento em pixels / tamanho aparente / frames
# o tamanho da caixa compensa o fato de o carro perto da camera andar mais pixels
MIN_FRAMES = 15   # menos que isso o trecho é curto demais para decidir
WINDOW = 5        # compara o ponto com outro alguns frames a frente (reduz tremor da caixa)
SLOW_RATIO = 0.5  # lento se anda menos da metade da velocidade tipica do proprio video

with open("settings.json") as f:
    settings = json.load(f)

video_id = settings["video_id"]
output_dir = f"./output/{video_id}"
tracks_path = f"{output_dir}/tracks.pkl"

with open(tracks_path, "rb") as f:
    data = pickle.load(f)

with open(f"./config/{video_id}.json") as f:
    config = json.load(f)

def track_speed(points, window):
    arr = np.array([(p[0], p[1], p[2], p[3]) for p in points], dtype=float)
    if len(arr) <= window:
        return None
    speeds = []
    for i in range(len(arr) - window):
        dt = arr[i + window, 0] - arr[i, 0]
        if dt <= 0:
            continue
        dist = np.hypot(arr[i + window, 1] - arr[i, 1], arr[i + window, 2] - arr[i, 2])
        scale = max((arr[i, 3] + arr[i + window, 3]) / 2, 1.0)
        speeds.append(dist / dt / scale)
    if not speeds:
        return None
    return float(np.median(speeds))

measured = []
for tid, points in data["trajectories"].items():
    cls_name = Counter(p[4] for p in points).most_common(1)[0][0]
    speed = track_speed(points, WINDOW) if len(points) >= MIN_FRAMES else None
    measured.append({"id": str(tid), "class": cls_name, "n_frames": len(points), "speed": speed})

reference_speeds = [t["speed"] for t in measured if t["speed"] is not None]
reference = float(np.median(reference_speeds)) if reference_speeds else 0.0
threshold = SLOW_RATIO * reference

for t in measured:
    if t["speed"] is None:
        t["label"] = "short"
    elif t["speed"] < threshold:
        t["label"] = "slow"
    else:
        t["label"] = "normal"

measured.sort(key=lambda t: (t["label"] != "slow", t["speed"] if t["speed"] is not None else 1e9))

result = {"video_id": video_id,
          "fps": config.get("fps", 25),
          "min_frames": MIN_FRAMES,
          "window": WINDOW,
          "slow_ratio": SLOW_RATIO,
          "reference_speed": reference,
          "threshold": threshold,
          "n_slow": sum(t["label"] == "slow" for t in measured),
          "n_normal": sum(t["label"] == "normal" for t in measured),
          "n_short": sum(t["label"] == "short" for t in measured),
          "tracks": measured}

os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/slow.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"referencia {reference:.4f}  limiar {threshold:.4f}  (metade do fluxo deste video)")
print(f"lentos {result['n_slow']}  normais {result['n_normal']}  curtos {result['n_short']}")
print("lentos:")
for t in measured:
    if t["label"] != "slow":
        continue
    print(f"  id {t['id']:>4}  {t['class']:<6}  {t['n_frames']:4} frames  vel {t['speed']:.4f}")
print(f"salvo em {output_dir}/slow.json")
