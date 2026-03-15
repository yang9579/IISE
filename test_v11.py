"""Test v11 fixes on all key images."""
import cv2, numpy as np, sys
sys.path.insert(0, '/users/8/yang9579/Github/IISE')
from detect_boxes import detect_in_circle, get_circle_params

DATA = '/users/8/yang9579/Github/IISE/data/'
PROB = '/users/8/yang9579/Github/IISE/detection_results/all_days/problematic/'

tests = [
    # Feb 18 - missing upper box
    (PROB + '20260218_085505_006401_combined.jpg', '20260218_085505'),
    (PROB + '20260218_085609_006411_combined.jpg', '20260218_085609'),
    # Feb 16 - bbox alignment
    (DATA + '2026-02-16/20260216_012406_144307_combined.jpg', '20260216_012406'),
    (DATA + '2026-02-16/20260216_012414_144317_combined.jpg', '20260216_012414'),
    # Previously good
    (PROB + '20260209_092413_119025_combined.jpg', '20260209_092413'),
    (PROB + '20260209_092338_118975_combined.jpg', '20260209_092338'),
    (PROB + '20260205_000250_019588_combined.jpg', '20260205_000250'),
    (PROB + '20260217_000128_160918_combined.jpg', '20260217_000128'),
]

for img_path, label in tests:
    img = cv2.imread(img_path)
    if img is None:
        print(f"{label}: CANNOT READ")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = get_circle_params(gray.shape)
    per = []
    all_boxes = []
    for cx, cy, r in circles:
        boxes, _, _ = detect_in_circle(gray, cx, cy, r)
        per.append(boxes)
        all_boxes.extend(boxes)
    total = len(all_boxes)
    status = "OK" if total == 4 else "MISS"
    print(f"[{status}] {label}: {total}/4  (L={len(per[0])}, R={len(per[1])})")
    for side, boxes in zip(['L', 'R'], per):
        for x, y, w, h, sc in boxes:
            print(f"      {side}: ({x},{y},{w},{h}) sc={sc:.2f}")
