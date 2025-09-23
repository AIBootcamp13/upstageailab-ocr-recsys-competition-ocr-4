'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/data/make_seg_detector_data.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
'''

import cv2
import numpy as np
import pyclipper
import torch
from collections import OrderedDict


class DBCollateFN:
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.inference_mode = False

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        filenames = [item['image_filename'] for item in batch]
        inverse_matrix = [item['inverse_matrix'] for item in batch]

        collated_batch = OrderedDict(images=torch.stack(images, dim=0),
                                     image_filename=filenames,
                                     inverse_matrix=inverse_matrix,
                                     )

        if self.inference_mode:
            return collated_batch

        polygons = [item['polygons'] for item in batch]

        prob_maps = []
        thresh_maps = []
        for i, image in enumerate(images):
            # Probability map / Threshold map 생성
            segmentations = self.make_prob_thresh_map(image, polygons[i], filenames[i])
            prob_map_tensor = torch.tensor(segmentations['prob_map']).unsqueeze(0)
            thresh_map_tensor = torch.tensor(segmentations['thresh_map']).unsqueeze(0)
            prob_maps.append(prob_map_tensor)
            thresh_maps.append(thresh_map_tensor)

        collated_batch.update(polygons=polygons,
                              prob_maps=torch.stack(prob_maps, dim=0),
                              thresh_maps=torch.stack(thresh_maps, dim=0),
                              )

        return collated_batch

    def make_prob_thresh_map(self, image, polygons, filename):
        _, h, w = image.shape
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)

        for poly in polygons:
            # Calculate the distance and polygons
            poly = poly.astype(np.int32)
            # Polygon point가 3개 미만이라면 skip
            if poly.size < 3:
                continue

            # https://arxiv.org/pdf/1911.08947.pdf 참고
            L = cv2.arcLength(poly, True) + np.finfo(float).eps
            D = cv2.contourArea(poly) * (1 - self.shrink_ratio ** 2) / L
            pco = pyclipper.PyclipperOffset()
            pco.AddPaths(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            # Probability map 생성
            shrinked = pco.Execute(-D)
            for s in shrinked:
                shrinked_poly = np.array(s)
                cv2.fillPoly(prob_map, [shrinked_poly], 1.0)

            # Threshold map 생성
            dilated = pco.Execute(D)
            for d in dilated:
                dilated_poly = np.array(d)

                xmin = dilated_poly[:, 0].min()
                xmax = dilated_poly[:, 0].max()
                ymin = dilated_poly[:, 1].min()
                ymax = dilated_poly[:, 1].max()
                width = xmax - xmin + 1
                height = ymax - ymin + 1

                polygon = poly[0].copy()
                polygon[:, 0] = polygon[:, 0] - xmin
                polygon[:, 1] = polygon[:, 1] - ymin

                xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width),
                                     (height, width))
                ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1),
                                     (height, width))

                distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
                for i in range(polygon.shape[0]):
                    j = (i + 1) % polygon.shape[0]
                    absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
                    distance_map[i] = np.clip(absolute_distance / D, 0, 1)
                distance_map = distance_map.min(axis=0)

                xmin_valid = min(max(0, xmin), thresh_map.shape[1] - 1)
                xmax_valid = min(max(0, xmax), thresh_map.shape[1] - 1)
                ymin_valid = min(max(0, ymin), thresh_map.shape[0] - 1)
                ymax_valid = min(max(0, ymax), thresh_map.shape[0] - 1)

                thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                    1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                        xmin_valid - xmin:xmax_valid - xmax + width],                        # noqa
                    thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        # Normalize the threshold map
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        return OrderedDict(prob_map=prob_map, thresh_map=thresh_map)

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]) + np.finfo(float).eps

        denom = 2 * np.sqrt(square_distance_1 * square_distance_2) + np.finfo(float).eps
        cosin = (square_distance - square_distance_1 - square_distance_2) / denom
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
