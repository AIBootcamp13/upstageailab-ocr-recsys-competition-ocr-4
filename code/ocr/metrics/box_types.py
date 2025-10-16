'''
*****************************************************************************************
* 참고 논문:
* CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
* https://arxiv.org/pdf/2006.06244.pdf
*
* 출처 Repository:
* https://github.com/clovaai/CLEval/tree/master/cleval
*****************************************************************************************
'''

import abc
import math

from scipy.spatial import ConvexHull
import numpy as np
import Polygon as polygon3
from shapely.geometry import Point
from collections import namedtuple

MAX_FIDUCIAL_POINTS = 50


def get_midpoints(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def point_distance(p1, p2):
    distx = math.fabs(p1[0] - p2[0])
    disty = math.fabs(p1[1] - p2[1])
    return math.sqrt(distx * distx + disty * disty)


BoundingBox = namedtuple('BoundingBox', ('area',
                                         'length_parallel',
                                         'length_orthogonal',
                                         'rectangle_center',
                                         'unit_vector',
                                         'unit_vector_angle',
                                         'corner_points'
                                         )
                         )


def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = math.sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index + 1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def to_xy_coordinates(unit_vector_angle, point):
    # returns converted unit vector coordinates in x, y coordinates
    angle_orthogonal = unit_vector_angle + math.pi / 2
    return point[0] * math.cos(unit_vector_angle) + point[1] * math.cos(angle_orthogonal), \
           point[0] * math.sin(unit_vector_angle) + point[1] * math.sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    # Requires: center_of_rotation to be a 2d vector. ex: (1.56, -23.4)
    #           angle to be in radians
    #           points to be a list or tuple of points. ex: ((1.56, -23.4), (1.56, -23.4))
    # Effects: rotates a point cloud around the center_of_rotation point by angle
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = math.atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = math.sqrt(sum([d ** 2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * math.cos(diff_angle),
                           center_of_rotation[1] + diff_length * math.sin(diff_angle)))

    return rot_points


def rectangle_corners(rectangle):
    # Requires: the output of mon_bounding_rectangle
    # Effects: returns the corner locations of the bounding rectangle
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append(
                (rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                 rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'],
                         corner_points)


# use this function to find the listed properties of the minimum bounding box of a point cloud
def custom_MinAreaRect(points):
    # Requires: points to be a list or tuple of 2D points. ex: ((5, 2), (3, 4), (6, 8))
    #           needs to be more than 2 points
    # Effects:  returns a namedtuple that contains:
    #               area: area of the rectangle
    #               length_parallel: length of the side that is parallel to unit_vector
    #               length_orthogonal: length of the side that is orthogonal to unit_vector
    #               rectangle_center: coordinates of the rectangle center
    #                   (use rectangle_corners to get the corner points of the rectangle)
    #               unit_vector: direction of the length_parallel side. RADIANS
    #                   (it's orthogonal vector can be found with the orthogonal_vector function
    #               unit_vector_angle: angle of the unit vector
    #               corner_points: set that contains the corners of the rectangle

    assert len(points) > 2, 'More than two points required.'

    try:
        hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    except:
        print(f"[WARN] ConvexHull failed. points: {points}")
        return (0, 0), (0, 0), 0

    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = math.atan2(min_rectangle['unit_vector'][1],
                                                    min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                          min_rectangle['rectangle_center'])

    return (set(rectangle_corners(min_rectangle)), (min_rectangle['length_parallel'],
                                                    min_rectangle['length_orthogonal']),
            min_rectangle['unit_vector_angle'])


class Box(metaclass=abc.ABCMeta):
    def __init__(
        self,
        points,
        confidence,
        transcription,
        orientation=None,
        is_dc=None,
    ):
        self.points = points
        self.confidence = confidence
        self.transcription = transcription
        self.orientation = orientation
        self.is_dc = transcription == "###" if is_dc is None else is_dc

    @abc.abstractmethod
    def __and__(self, other) -> float:
        """Returns intersection between two objects"""
        pass

    @abc.abstractmethod
    def subtract(self, other):
        """polygon subtraction"""
        pass

    @abc.abstractmethod
    def center(self):
        pass

    @abc.abstractmethod
    def center_distance(self, other):
        """center distance between each box"""

    @abc.abstractmethod
    def diagonal_length(self) -> float:
        """Returns diagonal length for box-level"""
        pass

    @abc.abstractmethod
    def is_inside(self, x, y) -> bool:
        """Returns point (x, y) is inside polygon."""
        pass

    @abc.abstractmethod
    def make_polygon_obj(self):
        # TODO: docstring 좀 더 자세히 적기
        """Make polygon object to calculate for future"""
        pass

    @abc.abstractmethod
    def pseudo_character_center(self, *args) -> list:
        """get character level boxes for TedEval pseudo center"""
        pass


class QUAD(Box):
    """Points should be x1,y1,...,x4,y4 (8 points) format"""

    def __init__(
        self,
        points,
        confidence=0.0,
        transcription="",
        orientation=None,
        is_dc=None,
        scale=None,
    ):
        super().__init__(points, confidence, transcription, orientation, is_dc)
        self.polygon = self.make_polygon_obj()
        self.scale = scale
        if self.is_dc:
            self.transcription = "#" * self.pseudo_transcription_length()
        if self.transcription is None:
            self.transcription = "#" * self.pseudo_transcription_length()

        self._center = None
        self._area = None
        self._aspect_ratio = None
        self._diagonal_length = None

    def __and__(self, other) -> float:
        """Get intersection between two area"""
        poly_intersect = self.polygon & other.polygon
        if len(poly_intersect) == 0:
            return 0.0
        return poly_intersect.area()

    def subtract(self, other):
        self.polygon = self.polygon - other.polygon

    def center(self):
        if self._center is None:
            self._center = self.polygon.center()
        return self._center

    def center_distance(self, other):
        return point_distance(self.center(), other.center())

    def area(self):
        if self._area is None:
            self._area = self.polygon.area()
        return self._area

    def __or__(self, other):
        return self.polygon.area() + other.polygon.area() - (self & other)

    def make_polygon_obj(self):
        point_matrix = np.empty((4, 2), np.int32)
        point_matrix[0][0] = int(self.points[0])
        point_matrix[0][1] = int(self.points[1])
        point_matrix[1][0] = int(self.points[2])
        point_matrix[1][1] = int(self.points[3])
        point_matrix[2][0] = int(self.points[4])
        point_matrix[2][1] = int(self.points[5])
        point_matrix[3][0] = int(self.points[6])
        point_matrix[3][1] = int(self.points[7])
        return polygon3.Polygon(point_matrix)

    def aspect_ratio(self):
        if self._aspect_ratio is None:
            top_side = point_distance((self.points[0], self.points[1]),
                                    (self.points[2], self.points[3]))
            right_side = point_distance((self.points[2], self.points[3]),
                                        (self.points[4], self.points[5]))
            bottom_side = point_distance((self.points[4], self.points[5]),
                                        (self.points[6], self.points[7]))
            left_side = point_distance((self.points[6], self.points[7]),
                                    (self.points[0], self.points[1]))
            avg_hor = (top_side + bottom_side) / 2
            avg_ver = (right_side + left_side) / 2

            self._aspect_ratio = min(100., avg_hor / (avg_ver + np.finfo(np.float32).eps))

        return self._aspect_ratio

    def pseudo_transcription_length(self):
        return min(round(0.5 + (max(self.aspect_ratio(), 1 / (self.aspect_ratio() + np.finfo(np.float32).eps)))), 100)

    def pseudo_character_center(self, vertical_aspect_ratio_threshold):
        chars = list()
        length = len(self.transcription)
        aspect_ratio = self.aspect_ratio()

        if length == 0:
            return chars

        if aspect_ratio >= vertical_aspect_ratio_threshold:
            left_top = self.points[0], self.points[1]
            right_top = self.points[2], self.points[3]
            right_bottom = self.points[4], self.points[5]
            left_bottom = self.points[6], self.points[7]
        else:
            left_top = self.points[6], self.points[7]
            right_top = self.points[0], self.points[1]
            right_bottom = self.points[2], self.points[3]
            left_bottom = self.points[4], self.points[5]

        p1 = get_midpoints(left_top, left_bottom)
        p2 = get_midpoints(right_top, right_bottom)

        unit_x = (p2[0] - p1[0]) / length
        unit_y = (p2[1] - p1[1]) / length

        for i in range(length):
            x = p1[0] + unit_x / 2 + unit_x * i
            y = p1[1] + unit_y / 2 + unit_y * i
            chars.append((x, y))
        return chars

    def diagonal_length(self) -> float:
        if self._diagonal_length is None:
            left_top = self.points[0], self.points[1]
            right_top = self.points[2], self.points[3]
            right_bottom = self.points[4], self.points[5]
            left_bottom = self.points[6], self.points[7]
            diag1 = point_distance(left_top, right_bottom)
            diag2 = point_distance(right_top, left_bottom)
            self._diagonal_length = (diag1 + diag2) / 2
        return self._diagonal_length

    def is_inside(self, x, y) -> bool:
        return self.polygon.isInside(x, y)


class POLY(Box):
    """Points should be x1,y1,...,xn,yn (2*n points) format"""

    def __init__(self, points, confidence=0.0, transcription="", orientation=None, is_dc=None):
        super().__init__(points, confidence, transcription, orientation, is_dc)
        self.num_points = len(self.points) // 2
        self.polygon = self.make_polygon_obj()
        self._aspect_ratio = self.make_aspect_ratio()
        if self.is_dc:
            self.transcription = "#" * self.pseudo_transcription_length()
        if self.transcription is None:
            self.transcription = "#" * self.pseudo_transcription_length()

        self._area = None
        self._center = None
        self._distance_idx_max_order = None
        self._pseudo_character_center = None

    def __and__(self, other) -> float:
        """Get intersection between two area"""
        poly_intersect = self.polygon & other.polygon
        if len(poly_intersect) == 0:
            return 0.0
        return poly_intersect.area()

    def subtract(self, other):
        """get substraction"""
        self.polygon = self.polygon - other.polygon

    def __or__(self, other):
        return self.polygon.area() + other.polygon.area() - (self & other)

    def area(self):
        if self._area is None:
            self._area = self.polygon.area()
        return self._area

    def center(self):
        if self._center is None:
            self._center = self.polygon.center()
        return self._center

    def center_distance(self, other):
        return point_distance(self.center(), other.center())

    def diagonal_length(self):
        left_top = self.points[0], self.points[1]
        right_top = self.points[self.num_points - 2], self.points[self.num_points - 1]
        right_bottom = self.points[self.num_points], self.points[self.num_points + 1]
        left_bottom = (
            self.points[self.num_points * 2 - 2],
            self.points[self.num_points * 2 - 1],
        )

        diag1 = point_distance(left_top, right_bottom)
        diag2 = point_distance(right_top, left_bottom)

        return (diag1 + diag2) / 2

    def is_inside(self, x, y) -> bool:
        return self.polygon.isInside(x, y)

    def check_corner_points_are_continuous(self, lt, rt, rb, lb):
        counter = 0
        while lt != rt:
            lt = (lt + 1) % self.num_points
            counter += 1

        while rb != lb:
            rb = (rb + 1) % self.num_points
            counter += 1

        return True

    def get_four_max_distance_from_center(self):
        if self._distance_idx_max_order is None:
            center_x, center_y = self.center()
            distance_from_center = list()
            point_x = self.points[0::2]
            point_y = self.points[1::2]

            for px, py in zip(point_x, point_y):
                distance_from_center.append(point_distance((center_x, center_y), (px, py)))

            distance_idx_max_order = np.argsort(distance_from_center)[::-1]
            self._distance_idx_max_order = distance_idx_max_order[:4]
        return self._distance_idx_max_order

    def make_polygon_obj(self):
        point_matrix = np.array(
            [[self.points[i], self.points[i + 1]] for i in range(0, len(self.points), 2)],
            dtype=np.int32)
        return polygon3.Polygon(point_matrix)

    def aspect_ratio(self):
        return self._aspect_ratio

    def pseudo_transcription_length(self):
        return min(round(0.5 + (max(self._aspect_ratio, 1 / (self._aspect_ratio + np.finfo(np.float32).eps)))), 100)

    def make_aspect_ratio(self):
        np.array(np.reshape(self.points, [-1, 2]))
        rect = custom_MinAreaRect(np.array(np.reshape(self.points, [-1, 2]), dtype=np.float32))
        width = rect[1][0]
        height = rect[1][1]

        return min(100., height / (width + np.finfo(np.float32).eps))

    def pseudo_character_center(self, vertical_aspect_ratio_threshold):
        if self._pseudo_character_center is None:
            chars = list()
            length = len(self.transcription)

            # Prepare polygon line estimation with interpolation
            point_x = self.points[0::2]
            point_y = self.points[1::2]
            points_x_top = point_x[: self.num_points // 2]
            points_x_bottom = point_x[self.num_points // 2 :]
            points_y_top = point_y[: self.num_points // 2]
            points_y_bottom = point_y[self.num_points // 2 :]

            # reverse bottom point order from left to right
            points_x_bottom = points_x_bottom[::-1]
            points_y_bottom = points_y_bottom[::-1]

            num_interpolation_section = (self.num_points // 2) - 1
            num_points_to_interpolate = length

            new_point_x_top, new_point_x_bottom = list(), list()
            new_point_y_top, new_point_y_bottom = list(), list()

            for sec_idx in range(num_interpolation_section):
                start_x_top, end_x_top = points_x_top[sec_idx], points_x_top[sec_idx + 1]
                start_y_top, end_y_top = points_y_top[sec_idx], points_y_top[sec_idx + 1]
                start_x_bottom, end_x_bottom = (
                    points_x_bottom[sec_idx],
                    points_x_bottom[sec_idx + 1],
                )
                start_y_bottom, end_y_bottom = (
                    points_y_bottom[sec_idx],
                    points_y_bottom[sec_idx + 1],
                )

                diff_x_top = (end_x_top - start_x_top) / num_points_to_interpolate
                diff_y_top = (end_y_top - start_y_top) / num_points_to_interpolate
                diff_x_bottom = (end_x_bottom - start_x_bottom) / num_points_to_interpolate
                diff_y_bottom = (end_y_bottom - start_y_bottom) / num_points_to_interpolate

                new_point_x_top.append(start_x_top)
                new_point_x_bottom.append(start_x_bottom)
                new_point_y_top.append(start_y_top)
                new_point_y_bottom.append(start_y_bottom)

                for num_pt in range(1, num_points_to_interpolate):
                    new_point_x_top.append(int(start_x_top + diff_x_top * num_pt))
                    new_point_x_bottom.append(int(start_x_bottom + diff_x_bottom * num_pt))
                    new_point_y_top.append(int(start_y_top + diff_y_top * num_pt))
                    new_point_y_bottom.append(int(start_y_bottom + diff_y_bottom * num_pt))
            new_point_x_top.append(points_x_top[-1])
            new_point_y_top.append(points_y_top[-1])
            new_point_x_bottom.append(points_x_bottom[-1])
            new_point_y_bottom.append(points_y_bottom[-1])

            len_section_for_single_char = (len(new_point_x_top) - 1) / len(self.transcription)

            for c in range(len(self.transcription)):
                center_x = (
                    new_point_x_top[int(c * len_section_for_single_char)]
                    + new_point_x_top[int((c + 1) * len_section_for_single_char)]
                    + new_point_x_bottom[int(c * len_section_for_single_char)]
                    + new_point_x_bottom[int((c + 1) * len_section_for_single_char)]
                ) / 4

                center_y = (
                    new_point_y_top[int(c * len_section_for_single_char)]
                    + new_point_y_top[int((c + 1) * len_section_for_single_char)]
                    + new_point_y_bottom[int(c * len_section_for_single_char)]
                    + new_point_y_bottom[int((c + 1) * len_section_for_single_char)]
                ) / 4

                chars.append((center_x, center_y))
            self._pseudo_character_center = chars
        return self._pseudo_character_center
