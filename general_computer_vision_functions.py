# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

# PARAMETER_DICT contains parameter names and values used by different functions
# provided by cv2. In the current version of this file, only compute_canny_image
# uses PARAMETER_DICT, but this may change in future versions.

PARAMETER_DICT = {'inverted_binary_threshold': 220,
                  'histogram_binary_ratio': 0.1,
                  'gamma_coefficient': 1.0,
                  'canny_hysterisis_threshold_min': 50,
                  'canny_hysterisis_threshold_max': 200,
                  'canny_aperture_size': 3,
                  'canny_l2_gradient': False,
                  'hough_distance_resolution': 1,
                  'hough_angle_resolution': 1,
                  'hough_threshold': 50,
                  'hough_minimum_line_length': 50,
                  'hough_maximum_gap_size': 4,
                  'alt_hough_distance_resolution': 1,
                  'alt_hough_angle_resolution': 1,
                  'alt_hough_threshold': 350,
                  'alt_hough_distance_resolution_divisor': 0,
                  'alt_hough_angle_resolution_divisor': 0,
                  'alt_hough_minimum_angle': 0,
                  'alt_hough_maximum_angle': 180,
                  'sift_num_features': 0,
                  'sift_num_octave_layers': 3,
                  'sift_contrast_threshold': 0.04,
                  'sift_edge_threshold': 10,
                  'sift_sigma': 1.6,
                  'sift_enable_precise_upscale': False}

# A binarization method often needs at least one user-provided parameter. The
# so-called triangle and Otsu methods need no user-provided parameters, so we
# prefer to use them. According to tests, the Otsu method works well in the
# logbook case.

def triangle_or_otsu_binarization(image, otsu_mode):
    mode = cv.THRESH_OTSU if otsu_mode else cv.THRESH_TRIANGLE
    binary_image = cv.threshold(image, 255, 255, mode)[1]
    binary_image = np.invert(binary_image)
    return binary_image

# The Canny algorithm is a classical method for detecting edges in an image.
# This algorithm is not used in the current version of the main algorithm, but
# we still include it in this file in order to give yet another example of a
# relevant function provided by cv2.

def compute_canny_image(image):
    threshold_min = PARAMETER_DICT['canny_hysterisis_threshold_min']
    threshold_max = PARAMETER_DICT['canny_hysterisis_threshold_max']
    aperture_size = PARAMETER_DICT['canny_aperture_size']
    l2_gradient = PARAMETER_DICT['canny_l2_gradient']
    canny_image = cv.Canny(image,
                           threshold_min,
                           threshold_max,
                           None,
                           aperture_size,
                           l2_gradient)
    return canny_image

# The following function detects the connected components in a binarized input
# image. We employ a basic implementation provided by cv2.

# The return value of the function is a list of four objects. The objects are
# the following:

# 1) An integer N which gives the number of connected components. Since the
# background is regarded as a component, the number of actual connected
# components is N - 1.

# 2) A label matrix, i.e., an integer matrix of the same shape as the input
# image with values 0, 1, 2, ..., N - 1 indicating for each pixel the component
# containing the pixel. Naturally, the label 0 labels the background pixels.

# 3) A list that contains for each connected component the description of the
# minimal rectangle containing the component. For further details, see the
# function compute_connected_component_rectangles in this file.

# 4) A list giving the coordinates of the centroid of each of the components.

# The objects 3) and 4) are lists of length N such that the first item of
# each list corresponds to the background.

def compute_connected_component_parameters(image):
    connected_component_parameters = (
        cv.connectedComponentsWithStatsWithAlgorithm(image,
                                                     8,
                                                     cv.CV_16U,
                                                     cv.CCL_DEFAULT)
    )
    return connected_component_parameters

# For a connected component computed by compute_connected_component_parameters,
# the function below constructs a rectangle containing the component. Initially,
# the rectangle is the minimal rectangle containing the component, but if
# needed, the rectangle can be stretched leftwards, rightwards, upwards and/or
# downwards.

# Strictly speaking, this function is not a computer vision function but rather
# a geometric function. However, since the connection between this function and
# compute_connected_component_parameters is very strong, we include the code of
# this function in this file.

# Recall that the return value of compute_connected_component_parameters is a
# list of four objects, the third one of which describes the minimal rectangles
# containing the connected components.

# More specifically, the third object is a list of lists such that for each
# connected component there is a list of four objects describing the minimal
# rectangle containing the component.

# Even more specifically, the description of a minimal rectangle is of the form
# [x_top_left, y_top_left, rectangle_width, rectangle_height], where:
# x_top_left is the x-coordinate of the top-left corner point,
# y_top_left is the y-coordinate of the top-left corner point,
# rectangle_width is the width, i.e., the x-length of the rectangle,
# rectangle_height is the height, i.e., the y-length of the rectangle.

# Recall that compute_connected_component_parameters regards the background of
# the input image as a special case of a component. In the list of rectangle
# descriptions, the first item corresponds to the background.

# In technical terms, the code itself of the function is based on very
# elementary geometric and arithmetic considerations, and so we will not
# provide detailed comments.

def compute_connected_component_rectangles(image,
                                           connected_component_parameters,
                                           left_extra_length=0,
                                           right_extra_length=0,
                                           top_extra_length=0,
                                           bottom_extra_length=0):
    height = image.shape[0]
    width = image.shape[1]
    # The list of rectangle descriptions is the third item in the argument
    # connected_component_parameters, and the first item of the description
    # list corresponds to the background of the input image and should,
    # therefore, be omitted. This explains the [2][1:]-part of the following
    # expression.
    connected_component_rectangle_descriptions = (
        connected_component_parameters[2][1:]
    )
    connected_component_rectangles = []
    for description in connected_component_rectangle_descriptions:
        x_top_left = description[0]
        y_top_left = description[1]
        rectangle_width = description[2]
        rectangle_height = description[3]
        x_1 = x_top_left - left_extra_length
        y_1 = y_top_left - top_extra_length
        x_2 = x_top_left + rectangle_width + right_extra_length
        y_2 = y_top_left + rectangle_height + bottom_extra_length
        x_1 = max(0, x_1)
        y_1 = max(0, y_1)
        x_2 = min(width - 1, x_2)
        y_2 = min(height - 1, y_2)
        connected_component_rectangles.append([[x_1, y_1], [x_2, y_2]])
    return connected_component_rectangles

# The function below detects so-called contours in a binarized input image.

# A contour is basically a shape or the outline of a shape.

# The purpose of the technical arguments cv.RETR_TREE and cv.CHAIN_APPROX_NONE
# is to guarantee that the amount of information returned by the function is
# maximal.

# In addition to the contours themselves, the function returns also the
# so-called hierarchy of contours.

# (Contours form a hierarchy based on inclusion relations among the contours.
# In the current version of the code, contour hierarchies are not used for
# anything, and so the topic is not discussed in greater detail.)

def detect_contours(image):
    contours, hierarchy = cv.findContours(image,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    return contours, hierarchy

# The following function is used to filter the contours detected by
# detect_contours.

# In fact, this function is not used by the current version of the code, and
# the purpose of its presence in the code is to remind that contour filtering
# may be a useful step.

# This is another example of a function in this file which is geometric in
# nature and not a computer vision function. But due to the strong connection
# to detect_contours, we include the function in this file.

# In its current form, the function implements a simple filtering based on the
# size of the area bounded by a given contour.

def filter_contours(contours,
                    contour_area_upper_bound=-1,
                    contour_area_lower_bound=-1):
    filtered_contours = []
    for contour in contours:
        size_of_area_bounded_by_contour = cv.contourArea(contour)
        if contour_area_upper_bound > 0:
            if size_of_area_bounded_by_contour > contour_area_upper_bound:
                continue
        if contour_area_lower_bound > 0:
            if size_of_area_bounded_by_contour < contour_area_lower_bound:
                continue
        filtered_contours.append(contour)
    return filtered_contours
