import cv2
import numpy as np


# Draw GT and Detection box
def draw_boxes(image_path, det_polys, gt_polys=None,
               det_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=2):
    image = cv2.imread(image_path)

    # Draw GT Polygons
    if gt_polys is not None:
        cv2.putText(image, f"{len(gt_polys)} GTs",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, gt_color)
        for box in gt_polys:
            box = np.array(box).reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [box], True, gt_color, thickness=thickness + 1)

    # Draw Detected Polygons
    cv2.putText(image, f"{len(det_polys)} DETs",
                (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, det_color)
    for box in det_polys:
        box = np.array(box).reshape(-1, 2).astype(np.int32)
        cv2.polylines(image, [box], True, det_color, thickness=thickness)

    return image
