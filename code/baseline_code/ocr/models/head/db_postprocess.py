'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/structure/representers/
*               seg_detector_representer.py
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
from shapely.geometry import Polygon
import pyclipper
import torch


class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, use_polygon=False):
        self.min_size = 3                       # minimum size of text region
        self.thresh = thresh                    # threshold for binarization
        self.box_thresh = box_thresh            # threshold for text region proposals
        self.max_candidates = max_candidates    # max number of text region proposals
        self.use_polygon = use_polygon          # use polygon or box

    def represent(self, batch, _pred):
        """
        batch: a dict produced by dataloaders.
            images: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            inverse_matrix: Warp Perspective Matrix, with shape (3, 3) as NDArray[float32]
            filename: the original filenames of images.
        pred:
            prob_maps: text region segmentation map, with shape (N, 1, H, W)
        """
        assert 'images' in batch is not None, "images is required in batch"
        images = batch['images']

        # Use prob_maps if pred is a dict
        if isinstance(_pred, dict):
            assert 'prob_maps' in _pred is not None, "prob_maps is required in _pred"
            pred = _pred['prob_maps']
        else:
            pred = _pred

        assert 'inverse_matrix' in batch is not None, "inverse_matrix is required in batch"
        inverse_matrix = batch['inverse_matrix']

        # Binarize the prediction
        segmentation = self.binarize(pred)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            if self.use_polygon:
                # Get polygons from segmentation
                boxes, scores = self.polygons_from_bitmap(
                                        pred[batch_index],
                                        segmentation[batch_index],
                                        inverse_matrix=inverse_matrix[batch_index])
            else:
                # Get boxes from segmentation
                boxes, scores = self.boxes_from_bitmap(
                                        pred[batch_index],
                                        segmentation[batch_index],
                                        inverse_matrix=inverse_matrix[batch_index])
            # Append to batch
            boxes_batch.append(boxes)
            scores_batch.append(scores)

        return boxes_batch, scores_batch

    @staticmethod
    def __transform_coordinates(coords, matrix):
        """
        Transform coordinates according to the warp matrix

        coords: (N, 2) as NDArray[float32]
        matrix: (3, 3) as NDArray[float32]
        return: (N, 2) as NDArray[float32]
        """
        coords = np.array(coords)
        coords = np.dot(matrix, np.vstack([coords.T, np.ones(coords.shape[0])]))
        coords /= coords[2, :]
        return coords.T[:, :2]

    def binarize(self, pred):
        # Binarize the prediction
        return pred > torch.Tensor([self.thresh]).to(device=pred.device)

    def polygons_from_bitmap(self, pred, _bitmap,
                             inverse_matrix=None):
        """
        Extracts polygons and their scores from a bitmap image.

        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]

        boxes = []
        scores = []

        # Find contours from the binarized map
        # contours: a list of contours
        # https://docs.opencv.org/4.9.0/d4/d73/tutorial_py_contours_begin.html
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Get the top N contours
        for contour in contours[:self.max_candidates]:
            # Approximate the contour with Douglas-Peucker algorithm
            # https://docs.opencv.org/4.9.0/dc/dcf/tutorial_js_contour_features.html
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue

            # Get the score of the box
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            # Unclip the box
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if box is None:
                    continue
            else:
                continue

            # Get the mini box
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            # Transform the coordinates
            box = self.__transform_coordinates(box, inverse_matrix)

            # Append to the list
            boxes.append(np.round(box).astype(np.int16).tolist())
            scores.append(score)

        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap,
                          inverse_matrix=None):
        """
        Extracts bounding boxes and their scores from a bitmap image.

        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]

        boxes = []
        scores = []

        # Find contours from the binarized map
        # contours: a list of contours
        # https://docs.opencv.org/4.9.0/d4/d73/tutorial_py_contours_begin.html
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)

        # Get the top N contours
        for index in range(num_contours):
            # Get the mini box
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            # Get the score of the box
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            # Unclip the box
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            # Transform the coordinates
            box = self.__transform_coordinates(box, inverse_matrix)

            # Append to the list
            boxes.append(np.round(box).astype(np.int16).tolist())
            scores.append(score)

        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        """
        Expands the given box by a specified ratio.

        box: a list of points of shape (N, 2)
        unclip_ratio: the ratio of unclipping the box
        return: a list of points of shape (N, 2)
        """

        # transform the box to polygon
        poly = Polygon(box)
        if poly.area == 0 or poly.length == 0:
            return None

        # get the expanded polygon
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)[0])

        return expanded

    def get_mini_boxes(self, contour):
        """
        Converts a contour into its minimum area bounding box.

        contour: a list of points of shape (N, 1, 2)
        return: a list of points of shape (N, 2)
        """

        # Get the bounding box
        # https://docs.opencv.org/4.9.0/de/d62/tutorial_bounding_rotated_ellipses.html
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        Calculates a score for a box in a bitmap.
        The score is the percentage of the box area that overlaps with
        the highlighted areas (marked as 1) in the bitmap.

        bitmap: a single map with shape (H, W), whose values are binarized as {0, 1}
        _box: a list of points of shape (N, 2)
        return: a score of the box as float32
        """

        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin

        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
