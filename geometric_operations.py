# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

# All of the global variables below are technical arguments used by the fitLine
# function of cv2 which is invoked once in the function compute_horizontal_or_
# vertical_lines_using_rectangles. We have given them the default values
# suggested by the cv2 documentation.

FIT_LINE_DISTANCE_PARAMETER = 0
FIT_LINE_RADIUS_EPS = 0.01
FIT_LINE_ANGLE_EPS = 0.01

# This is a simple function that filters rectangles based on the lengths of the
# sides of the rectangles.

def filter_rectangles(rectangles,
                      horizontal_length_upper_bound=-1,
                      horizontal_length_lower_bound=-1,
                      vertical_length_upper_bound=-1,
                      vertical_length_lower_bound=-1):
    filtered_rectangles = []
    for rectangle in rectangles:
        (x_1, y_1), (x_2, y_2) = rectangle
        if horizontal_length_upper_bound > 0:
            if np.abs(x_2 - x_1) > horizontal_length_upper_bound:
                continue
        if horizontal_length_lower_bound > 0:
            if np.abs(x_2 - x_1) < horizontal_length_lower_bound:
                continue
        if vertical_length_upper_bound > 0:
            if np.abs(y_2 - y_1) > vertical_length_upper_bound:
                continue
        if vertical_length_lower_bound > 0:
            if np.abs(y_2 - y_1) < vertical_length_lower_bound:
                continue
        filtered_rectangles.append(rectangle)
    return filtered_rectangles

# The task of the function below is to compute a point on a line determined by
# a given line element.

# The argument line_element is of the form [delta_x, delta_y, x_0, y_0] and it
# determines the line satisfying the following equation:
# y - y_0 = (delta_y / delta_x) * (x - x_0)

# The goal is to determine the point [x_target, y_target] contained in this
# line. The idea is that either x_target or y_target has a numerical value
# (i.e., is not None) as a function argument, and the function is used to
# compute a numerical value for the other variable.

# More specifically, if x_target is not None, the value of y_target is given by
# y_target = (delta_y / delta_x) * (x_target - x_0) + y_0
# On the other hand, if y_target is not None, the value of x_target is given by
# x_target = (delta_x / delta_y) * (y_target - y_0) + x_0

def compute_line_point_using_line_element(line_element,
                                          x_target=None,
                                          y_target=None):
    delta_x, delta_y, x_0, y_0 = line_element
    if x_target is not None:
        y_target = (delta_y / delta_x) * (x_target - x_0) + y_0
        return y_target
    if y_target is not None:
        x_target = (delta_x / delta_y) * (y_target - y_0) + x_0
        return x_target

# The following function is the main function of this file. It is invoked by
# the main algorithm when the actual horizontal/vertical table lines are
# computed.

# The setting is the following: We are given a list of rectangles, and the main
# assumption is that each one of the rectangles contains a table line. The task
# is to fit a line segment to the line-like pixels in each rectangle.

# See the code itself for a more detailed account on the steps of the algorithm.

# Note that, in this context, the function argument image is a binarized image
# whose non-zero pixels are exactly the line-like pixels which were determined
# earlier by the main algorithm.

def compute_horizontal_or_vertical_lines_using_rectangles(image,
                                                          rectangles,
                                                          compute_horizontal):
    horizontal_or_vertical_lines = []
    for rectangle in rectangles:
        (x_1_rect, y_1_rect), (x_2_rect, y_2_rect) = rectangle
        cropped_image = image[y_1_rect:y_2_rect + 1, x_1_rect:x_2_rect + 1]
        # 1) Determine the coordinates of all of the line-like pixels contained
        # in the given rectangle. These coordinates are listed in the numpy
        # array non_zero_points. Note that this determination of coordinates is
        # done in a cropped image, so in order to get the final result in the
        # coordinate system of the original input image, there has to be a
        # coordinate transformation later, see *).
        non_zero_indices = np.nonzero(cropped_image)
        non_zero_x_indices = non_zero_indices[1]
        non_zero_y_indices = non_zero_indices[0]
        # Grab along the extremal values of the coordinates for future use.
        x_min_ind = min(non_zero_x_indices)
        x_max_ind = max(non_zero_x_indices)
        y_min_ind = min(non_zero_y_indices)
        y_max_ind = max(non_zero_y_indices)
        # Recall that cv2 uses the [x, y]-convention for coordinates and not
        # the [y, x]-convention used in the context of matrices. This explains
        # why non_zero_points is constructed in the following way.
        non_zero_points = np.array([non_zero_x_indices, non_zero_y_indices],
                                   np.float32).transpose()
        # 2) Fit a line to the line-like pixels contained in the rectangle. We
        # use a standard function provided by cv2. Recall that all of the
        # global variables used here are default technical arguments. The
        # meaning of cv.DIST_L2 is that a standard least-squares computation is
        # used in the fitting.
        line_element = cv.fitLine(non_zero_points,
                                  cv.DIST_L2,
                                  FIT_LINE_DISTANCE_PARAMETER,
                                  FIT_LINE_RADIUS_EPS,
                                  FIT_LINE_ANGLE_EPS)
        # 3) Since the return value of 2) is a line element, we need to use
        # compute_line_point_using_line_element (and the extremal coordinate
        # values) to compute the endpoints of the line segment representing
        # the relevant table line. In the case of a horizontal table line, we
        # know the values of the x-coordinates of the endpoints and need to
        # compute the y-coordinates. In the case of a vertical table line, the
        # roles of the coordinate axes are switched.
        delta_x, delta_y, x_0, y_0 = line_element
        if compute_horizontal:
            x_1 = x_min_ind
            x_2 = x_max_ind
            if delta_y == 0:
                y_1 = y_0
                y_2 = y_0
            else:
                y_1 = compute_line_point_using_line_element(line_element,
                                                            x_target=x_1)
                y_2 = compute_line_point_using_line_element(line_element,
                                                            x_target=x_2)
        else:
            y_1 = y_min_ind
            y_2 = y_max_ind
            if delta_x == 0:
                x_1 = x_0
                x_2 = x_0
            else:
                x_1 = compute_line_point_using_line_element(line_element,
                                                            y_target=y_1)
                x_2 = compute_line_point_using_line_element(line_element,
                                                            y_target=y_2)
        # *) Perform a coordinate transformation in order to express the final
        # result in the coordinate system of the original input image.
        x_1 = int(x_1 + x_1_rect)
        y_1 = int(y_1 + y_1_rect)
        x_2 = int(x_2 + x_1_rect)
        y_2 = int(y_2 + y_1_rect)
        horizontal_or_vertical_lines.append([[x_1, y_1], [x_2, y_2]])
    return horizontal_or_vertical_lines
