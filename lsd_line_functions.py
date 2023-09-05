# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

# The following function is a simple (and redundant) function used to study
# lsd_lines. The current version performs a simple study of the angle
# attribute of an lsd_line object.

def study_lsd_lines(lsd_lines):
    min_angle = min([lsd_line.angle for lsd_line in lsd_lines])
    max_angle = max([lsd_line.angle for lsd_line in lsd_lines])
    assert -(np.pi + 0.01) <= min_angle <= np.pi + 0.01
    assert -(np.pi + 0.01) <= max_angle <= np.pi + 0.01

# The simple function below uses attributes of an lsd_line to determine its
# start point and end point.

def get_lsd_line_start_point_and_end_point(lsd_line):
    start_point = (int(lsd_line.startPointX),
                   int(lsd_line.startPointY))
    end_point = (int(lsd_line.endPointX),
                 int(lsd_line.endPointY))
    return start_point, end_point

# The following function is used to display pairs of line segments consisting
# of an lsd_line (an object with a number of methods and attributes) and a
# normal line (a pair of two endpoints).

# The idea is that we have a collection of lsd_lines whose elements are
# transformed in some way (e.g. by a geometric transformation) and the results
# are represented as normal lines. This function takes each pair consisting of
# the original lsd_line and the transformed line and displays the pair in an
# image.

# Since the function is not essential from the point of view of the main
# algorithm, we will not explain its details here. See the discussion on
# gui_functions.py for details as to how functions in cv2 can be used to
# display images.

def study_lsd_lines_and_simple_lines(image,
                                     lsd_lines,
                                     lines,
                                     lsd_color,
                                     color,
                                     lsd_thickness,
                                     thickness,
                                     margin_size):
    num_lsd_lines = len(lsd_lines)
    num_lines = len(lines)
    assert num_lsd_lines == num_lines
    for lsd_line, line in zip(lsd_lines, lines):
        line_pair_image = np.zeros_like(cv.cvtColor(image, cv.COLOR_GRAY2BGR))
        (lsd_x_1, lsd_y_1), (lsd_x_2, lsd_y_2) = (
            get_lsd_line_start_point_and_end_point(lsd_line)
        )
        cv.line(line_pair_image,
                (lsd_x_1, lsd_y_1),
                (lsd_x_2, lsd_y_2),
                lsd_color,
                lsd_thickness,
                cv.LINE_AA)
        (x_1, y_1), (x_2, y_2) = line
        cv.line(line_pair_image,
                (x_1, y_1),
                (x_2, y_2),
                color,
                thickness,
                cv.LINE_AA)
        height, width, _ = line_pair_image.shape
        x_min = min(x_1, x_2, lsd_x_1, lsd_x_2)
        x_max = max(x_1, x_2, lsd_x_1, lsd_x_2)
        y_min = min(y_1, y_2, lsd_y_1, lsd_y_2)
        y_max = max(y_1, y_2, lsd_y_1, lsd_y_2)
        x_min = max(0, x_min - margin_size)
        x_max = min(width, x_max + margin_size + 1)
        y_min = max(0, y_min - margin_size)
        y_max = min(height, y_max + margin_size + 1)
        cropped_image = line_pair_image[y_min:y_max, x_min:x_max]
        cv.imshow('Line pair image', cropped_image)
        cv.waitKey(0)

# The simple function below computes the length of an lsd_line.

# Note that an lsd_line does have the attribute lineLength, but this attribute
# does not necessarily give the right length with respect to the coordinate
# system of the original input image (we omit the details as to why this can
# happen).

def compute_lsd_line_length(lsd_line):
    x_1, y_1 = lsd_line.getStartPoint()
    x_2, y_2 = lsd_line.getEndPoint()
    length = np.sqrt(np.square(x_2 - x_1) + np.square(y_2 - y_1))
    return length

# The function below is used by the main algorithm to filter lsd_lines. In the
# context of the main algorithm, an lsd_line is filtered if it is too short or
# not horizontal/vertical enough.

# An lsd_line is horizontal or vertical enough if np.abs(np.sin(lsd_line.angle))
# or np.abs(np.cos(lsd_line.angle)) is small enough, respectively.

# The full function provides also other filtering conditions. The actual code
# of the function is rather self-explanatory, so we will not consider it in
# great detail.

def filter_lsd_lines(lsd_lines,
                     length_upper_bound=-1,
                     length_lower_bound=-1,
                     cos_upper_bound=-1,
                     cos_lower_bound=-1,
                     sin_upper_bound=-1,
                     sin_lower_bound=-1):
    filtered_lsd_lines = []
    for lsd_line in lsd_lines:
        if length_upper_bound > 0:
            length = compute_lsd_line_length(lsd_line)
            if length > length_upper_bound:
                continue
        if length_lower_bound > 0:
            length = compute_lsd_line_length(lsd_line)
            if length < length_lower_bound:
                continue
        if cos_upper_bound > 0:
            if np.abs(np.cos(lsd_line.angle)) > cos_upper_bound:
                continue
        if cos_lower_bound > 0:
            if np.abs(np.cos(lsd_line.angle)) < cos_lower_bound:
                continue
        if sin_upper_bound > 0:
            if np.abs(np.sin(lsd_line.angle)) > sin_upper_bound:
                continue
        if sin_lower_bound > 0:
            if np.abs(np.sin(lsd_line.angle)) < sin_lower_bound:
                continue        
        filtered_lsd_lines.append(lsd_line)
    return filtered_lsd_lines

# The purpose of the function below is to transform a collection of lsd_lines
# so that if lsd_line is an element of the collection which is nearly horizontal
# or vertical, then lsd_line is mapped to a line that goes through the middle
# point of lsd_line and is exactly horizontal or vertical, respectively.

# We mention that the pt attribute of an lsd_line gives the coordinates of the
# middle point of the lsd_line, but we will not discuss other details of this
# functions code, since the function is not used in the current version of the
# main algorithm.

def lsd_lines_to_horizontal_or_vertical_lines(lsd_lines, to_horizontal):
    horizontal_or_vertical_lines = []
    for lsd_line in lsd_lines:
        x_mid, y_mid = lsd_line.pt
        half_length = compute_lsd_line_length(lsd_line) / 2
        if to_horizontal:
            x_1 = int(x_mid - half_length)
            y_1 = int(y_mid)
            x_2 = int(x_mid + half_length)
            y_2 = int(y_mid)
        else:
            x_1 = int(x_mid)
            y_1 = int(y_mid - half_length)
            x_2 = int(x_mid)
            y_2 = int(y_mid + half_length)
        horizontal_or_vertical_lines.append([[x_1, y_1], [x_2, y_2]])
    return horizontal_or_vertical_lines

# The simple function below can be used to draw lsd_lines in an image. It
# invokes the rather self-explanatory function line of cv2.

# The only potentially unintuitive argument is the constant cv.LINE_AA. Its
# purpose is to make the drawn lines antialiased.

def draw_lsd_lines(image, lsd_lines, color, thickness):
    for lsd_line in lsd_lines:
        start_point, end_point = (
            get_lsd_line_start_point_and_end_point(lsd_line)
        )
        cv.line(image,
                start_point,
                end_point,
                color,
                thickness,
                cv.LINE_AA)

# The following function is another function for drawing lsd_lines. It is not
# used by the current version of the code.

# The function refers to the technical attribute octave of lsd_lines. From the
# point of view of the main algorithm, the exact nature of octaves is
# unimportant, so we will not discuss the matter in detail.

def draw_lsd_lines_in_octave_images(image,
                                    lsd_lines,
                                    color,
                                    thickness,
                                    num_octaves):
    octave_images = []
    for octave in range(-1, num_octaves + 1):
        octave_image = image.copy()
        for lsd_line in lsd_lines:
            if (octave == -1) or (octave == lsd_line.octave):
                start_point, end_point = (
                    get_lsd_line_start_point_and_end_point(lsd_line)
                )
                cv.line(octave_image,
                        start_point,
                        end_point,
                        color,
                        thickness,
                        cv.LINE_AA)
        octave_images.append(octave_image)
    return octave_images
