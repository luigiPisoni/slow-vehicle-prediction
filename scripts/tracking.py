from ultralytics.models.yolo import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import numpy as np
import cv2
import glob
import json
import os
import pickle

with open("settings.json") as f:
    settings = json.load(f)

model = YOLO(settings["yolo_model"])
tracker = DeepSort(max_age=30)  # n frames de oclusao

video_id = settings["video_id"]
path = settings["dataset_dir"] + video_id

image_paths = sorted(glob.glob(path + "/*.jpg"))

# --- config (editada a mao) e saida (gerada pelo codigo) ---
config_path = f"./config/{video_id}.json"
output_dir = f"./output/{video_id}"
os.makedirs("./config", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

first_frame = cv2.imread(image_paths[0])
frame_height, frame_width = first_frame.shape[:2]

# cria a config padrao se ainda nao existir (sem zonas)
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        json.dump({"video_id": video_id,
                   "fps": 25,
                   "frame_size": [frame_width, frame_height],
                   "exclusion_zones": []
                }, f, indent=2)
    print(f"Config criada em {config_path} (sem zonas de exclusao)")

with open(config_path) as f:
    config = json.load(f)

# cada zona e um retangulo axis-aligned (4 cantos no json)
rects = []
for z in config["exclusion_zones"]:
    pts = np.array(z, dtype=np.float32)
    rects.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                  float(pts[:, 0].max()), float(pts[:, 1].max())))
zones = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
         for x1, y1, x2, y2 in rects]
print(f"{len(zones)} zonas de exclusao carregadas")

def in_exclusion(cx, cy):
    # True se o ponto cai dentro (ou na borda) de qualquer retangulo
    return any(x1 <= cx <= x2 and y1 <= cy <= y2
               for x1, y1, x2, y2 in rects)

trajectories = defaultdict(list)  # tid -> [(frame_idx, cx, cy, size, cls_name)]
frame_boxes = {}                  # frame_idx -> [(tid, x1, y1, x2, y2, cls_name)], usado depois para desenhar

for frame_idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)

    # classes habilitadas: carro, motocicleta, onibus e caminhao, respectivamente
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # coordenadas da bounding box
        conf = float(box.conf[0])              # confianca da predicao
        cls_id = int(box.cls[0])               # classe da predicao

        # zonas de exclusao: descarta se a base da caixa esta numa zona
        if in_exclusion((x1 + x2) / 2, y2):
            continue

        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, model.names[cls_id]))

    tracks = tracker.update_tracks(detections, frame=frame)

    frame_boxes[frame_idx] = []
    for track in tracks:
        # confirmada E casada com uma deteccao neste frame (ignora previsao do Kalman)
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()  # left top right bottom
        cls_name = track.get_det_class()

        cx, cy = (x1 + x2) / 2, y2                 # centro da base da caixa (ponto no chao)
        size = ((x2 - x1) * (y2 - y1)) ** 0.5      # proxy de escala (distancia da camera)

        trajectories[track_id].append((frame_idx, cx, cy, size, cls_name))
        frame_boxes[frame_idx].append((track_id, x1, y1, x2, y2, cls_name))

# salva para nao precisar rodar a deteccao de novo
with open(f"{output_dir}/tracks.pkl", "wb") as f:
    pickle.dump({"video_id": video_id,
                 "trajectories": dict(trajectories),
                 "frame_boxes": frame_boxes,
                 "frame_height": frame_height,
                 "image_paths": image_paths,
                 "zones": zones}, f)

# resumo da execucao (nao toca na config editada a mao)
lengths = [len(t) for t in trajectories.values()]
summary = {"video_id": video_id,
           "n_frames": len(image_paths),
           "n_tracks": len(lengths),
           "n_zones": len(zones),
           "length_p10_p50_p90": np.percentile(lengths, [10, 50, 90]).tolist() if lengths else None}
with open(f"{output_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"{len(trajectories)} trajetos salvos em {output_dir}")
# print(summary)