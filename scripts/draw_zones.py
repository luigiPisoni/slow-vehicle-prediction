import cv2, glob, json, os
import numpy as np

with open("settings.json") as f:
    settings = json.load(f)

video_id = settings["video_id"]
path = settings["dataset_dir"] + video_id
image_paths = sorted(glob.glob(path + "/*.jpg"))
config_path = f"./config/{video_id}.json"

base = cv2.imread(image_paths[0])
zones = []     # retangulos prontos (4 cantos)
origin = None  # primeiro canto do retangulo atual
cursor = None  # posicao do mouse (preview)

def rect_corners(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    xa, xb = min(x1, x2), max(x1, x2)
    ya, yb = min(y1, y2), max(y1, y2)
    return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]

def draw_rect(img, corners, fill=False):
    pts = np.array(corners, dtype=np.int32)
    if fill:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    for p in corners:
        cv2.circle(img, p, 4, (0, 255, 255), -1)
    return img

def redraw():
    img = base.copy()
    for z in zones:
        img = draw_rect(img, z, fill=True)
        img = draw_rect(img, z, fill=False)
    if origin is not None and cursor is not None:
        img = draw_rect(img, rect_corners(origin, cursor), fill=False)
    elif origin is not None:
        cv2.circle(img, origin, 4, (0, 255, 255), -1)
    return img

def on_mouse(event, x, y, flags, param):
    global origin, cursor
    cursor = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if origin is None:
            origin = (x, y)
        else:
            zones.append(rect_corners(origin, (x, y)))
            origin = None

cv2.namedWindow("Exclusao")
cv2.setMouseCallback("Exclusao", on_mouse)

# dois cliques = um retangulo,  u = desfaz, q = termina
while True:
    cv2.imshow("Exclusao", redraw())
    k = cv2.waitKey(20) & 0xFF
    if k == ord("u"):
        if origin is not None:
            origin = None
        elif zones:
            zones.pop()
    elif k == ord("q"):
        break
cv2.destroyAllWindows()

os.makedirs("./config", exist_ok=True)
config = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
config.update({"video_id": video_id,
               "fps": config.get("fps", 25),
               "frame_size": [base.shape[1], base.shape[0]],
               "exclusion_zones": [[list(p) for p in z] for z in zones]})
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

img = base.copy()
for z in zones:
    img = draw_rect(img, z, fill=True)
os.makedirs(f"./output/{video_id}", exist_ok=True)
cv2.imwrite(f"./output/{video_id}/zones_check.jpg", img)
print(f"{len(zones)} zonas salvas em {config_path}")
