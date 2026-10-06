from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import glob
import os


model = YOLO("yolov8l.pt")
tracker = DeepSort(max_age=30) # n frames de oclusao
output_dir = "./output"  
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob("./DETRAC-Images/MVI_20011/*.jpg"))

unique_vehicle_ids = set()

for path in image_paths:
    frame = cv2.imread(path)
    # classes habilitadas: carro, motocicleta, onibus e caminhao, respectivamente
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist() # coordenadas da bounding box
        conf = float(box.conf[0]) # confianca da predicao
        cls_id = int(box.cls[0]) # classe da predicao
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, model.names[cls_id]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb() # left top right bottom (x1, y1, x2, y2)
        cls_name = track.get_det_class()

        unique_vehicle_ids.add(track_id)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} ID:{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_name = output_dir + "/tracked_" + path.split("/")[-1]
    cv2.imwrite(out_name, frame)

print(f"Total unique vehicles tracked: {len(unique_vehicle_ids)}")