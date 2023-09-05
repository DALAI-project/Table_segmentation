# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

import general_computer_vision_functions
import geometric_operations
import lsd_line_functions
import utilities

# Most of the global variables are used only in the construction of images
# which describe the functioning and results of the main functions.

# The values of some of the technical variables, namely LSD_LINE_DETECTOR_SCALE,
# LSD_LINES_IMAGE_COLOR and LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR, are
# well-justified and should probably not be altered.

# It is perhaps a good idea to test varying values for LSD_LINES_IMAGE_THICKNESS
# and LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS, and it is quite likely
# that these variables should be function arguments instead of global variables.

LSD_LINE_DETECTOR_SCALE = 2

LSD_LINES_IMAGE_COLOR = 255
LSD_LINES_IMAGE_THICKNESS = 5

LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR = 255
LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS = 1

LSD_LINES_FULL_IMAGE_COLOR = (255, 0, 0)
LSD_LINES_FULL_IMAGE_THICKNESS = 5

RECTANGLE_COMPONENT_RECTANGLES_IMAGE_THICKNESS = 2

TABLE_LINE_COLOR = (255, 0, 0)
TABLE_LINE_THICKNESS = 5

TABLE_ELEMENT_RECTANGLE_COLOR = (255, 0, 0)
TABLE_ELEMENT_RECTANGLE_THICKNESS = 2

# Depending on whether the value of detect_horizontal_lines is True or False,
# the function below detects the horizontal or vertical table lines in the
# input image, respectively. See the comments in the code below for a more
# detailed description of the algorithm steps.

# The directory ./example_images/example_progress_images contains the progress
# image examples referred to in the code. We use the symbol k to refer to the
# image file image_k.jpg.

# The full list of progress image variable names is the following:
# [image,
#  lsd_lines_full_image,
#  lsd_lines_image,
#  lsd_lines_component_short_rectangles_image,
#  lsd_lines_component_short_rectangles_zeros_image,
#  lsd_lines_component_rectangles_image,
#  rectangle_component_rectangles_image,
#  rectangle_component_rectangles_zeros_image,
#  rectangle_component_filtered_rectangles_zeros_image,
#  rectangle_component_filtered_rectangles_image,
#  horizontal_or_vertical_table_lines_rectangles_image,
#  horizontal_or_vertical_table_lines_full_image]

def detect_horizontal_or_vertical_table_lines(image,
                                              lsd_lines,
                                              detect_horizontal_lines,
                                              line_length_lower_bound=-1,
                                              cos_upper_bound=-1,
                                              sin_upper_bound=-1,
                                              right_extra_length=0,
                                              bottom_extra_length=0,
                                              rectangle_length_lower_bound=-1,
                                              construct_progress_images=False):
    # 1) Filter lsd_lines which are not long enough or which are not 
    # sufficiently horizontal/vertical. The sin limit is used in the horizontal
    # case and the cos limit in the vertical case.
    lsd_lines = lsd_line_functions.filter_lsd_lines(
        lsd_lines,
        length_lower_bound=line_length_lower_bound,
        cos_upper_bound=cos_upper_bound,
        sin_upper_bound=sin_upper_bound
    )
    # 2) Draw the remaining lsd_lines in a zero-initialized image of the same
    # shape as the input image.
    # Relevant progress image variable names:
    # lsd_lines_full_image,
    # lsd_lines_image
    # Relevant progress image examples: 2, 3, 13, 14
    lsd_lines_image = np.zeros_like(image)
    lsd_line_functions.draw_lsd_lines(
        lsd_lines_image,
        lsd_lines,
        LSD_LINES_IMAGE_COLOR,
        LSD_LINES_IMAGE_THICKNESS
    )
    # 3) Determine the connected components in the image drawn in 2).
    lsd_lines_component_parameters = (
        general_computer_vision_functions
        .compute_connected_component_parameters(lsd_lines_image)
    )
    # 4) For each connected component determined in 3), we construct the minimal
    # rectangle containing said component and extend the rectangle either
    # rightwards or downwards, depending on whether we are looking for
    # horizontal or vertical table lines, respectively.
    # The idea is that for each relevant table line there are lsd_lines
    # close to it and, therefore, also connected components determined in 3) 
    # close to it. The hope is that, by extending the rectangles, each relevant 
    # table line will be contained in a collection of mutually-intersecting 
    # rectangles. (By definition, a collection of rectangles is said to be
    # mutually-intersecting if for each rectangle in the collection there is at
    # least one other rectangle in the collection which intersects the first
    # rectangle).
    # Relevant progress image variable names:
    # lsd_lines_component_short_rectangles_image,
    # lsd_lines_component_short_rectangles_zeros_image,
    # lsd_lines_component_rectangles_image,
    # Relevant progress image examples: 4, 5, 6, 15, 16, 17
    lsd_lines_component_rectangles = (
        general_computer_vision_functions
        .compute_connected_component_rectangles(
            image,
            lsd_lines_component_parameters,
            right_extra_length=right_extra_length,
            bottom_extra_length=bottom_extra_length
        )
    )
    # 5) Draw the rectangles constructed in 4) in a zero-initialized image of
    # the same shape as the input image. Determine first the connected
    # components of this image and then the minimal rectangles containing these
    # components. The hope is that every relevant horizontal/vertical table line
    # is contained in exactly one of the minimal rectangles. We also hope that 
    # the minimal rectangles which do not contain a relevant horizontal/
    # vertical table line are exactly those which are too short with respect to
    # the horizontal/vertical direction. (The lower bound for the rectangle
    # length is given by the function argument rectangle_length_lower_bound.)
    # Relevant progress image variable names:
    # lsd_lines_component_rectangles_image,
    # rectangle_component_rectangles_image,
    # rectangle_component_rectangles_zeros_image
    # Relevant progress image examples: 6, 7, 8, 17, 18, 19
    lsd_lines_component_rectangles_image = np.zeros_like(image)
    utilities.draw_rectangles(
        lsd_lines_component_rectangles_image,
        lsd_lines_component_rectangles,
        LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
        LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
    )
    rectangle_component_parameters = (
        general_computer_vision_functions
        .compute_connected_component_parameters(
            lsd_lines_component_rectangles_image
        )
    )
    rectangle_component_rectangles = (
        general_computer_vision_functions
        .compute_connected_component_rectangles(
            image,
            rectangle_component_parameters
        )
    )
    # 6) Remove those minimal rectangles constructed in 5) which are too short
    # in the horizontal/vertical direction. If the algorithm works as it is
    # supposed to, there is an exact correspondence between the remaining
    # minimal rectangles and the relevant horizontal/vertical table lines.
    # Relevant progress image variable name:
    # rectangle_component_filtered_rectangles_zeros_image
    # Relevant progress image examples: 9, 20
    if detect_horizontal_lines:
        rectangle_component_rectangles = (
            geometric_operations.filter_rectangles(
                rectangle_component_rectangles,
                horizontal_length_lower_bound=rectangle_length_lower_bound
            )
        )
    else:
        rectangle_component_rectangles = (
            geometric_operations.filter_rectangles(
                rectangle_component_rectangles,
                vertical_length_lower_bound=rectangle_length_lower_bound
            )
        )
    # 7) For each of the remaining minimal rectangles, construct the line
    # segment that is the best fit to the original line-like pixels contained
    # in the rectangle. This line segment will be the ultimate horizontal/
    # vertical table line returned by this function.
    # Relevant progress image variable names:
    # rectangle_component_filtered_rectangles_image,
    # horizontal_or_vertical_table_lines_rectangles_image,
    # horizontal_or_vertical_table_lines_full_image
    # Relevant progress image examples: 10, 11, 12, 21, 22, 23
    horizontal_or_vertical_table_lines = (
        geometric_operations
        .compute_horizontal_or_vertical_lines_using_rectangles(
            lsd_lines_image,
            rectangle_component_rectangles,
            detect_horizontal_lines
        )
    )
    if construct_progress_images:
        # We have already pointed out which progress images are relevant to
        # a given step of the algorithm, and so the comments will be minimal
        # from here onwards. We will mainly just point out which progress image
        # examples are relevent to a given variable. For example:
        # lsd_lines_full_image: 2, 13
        lsd_lines_full_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        lsd_line_functions.draw_lsd_lines(
            lsd_lines_full_image,
            lsd_lines,
            LSD_LINES_FULL_IMAGE_COLOR,
            LSD_LINES_FULL_IMAGE_THICKNESS
        )
        # lsd_lines_component_short_rectangles_image: 4, 15
        lsd_lines_component_short_rectangles = (
            general_computer_vision_functions
            .compute_connected_component_rectangles(
                image,
                lsd_lines_component_parameters,
            )
        )
        lsd_lines_component_short_rectangles_image = lsd_lines_image.copy()
        utilities.draw_rectangles(
            lsd_lines_component_short_rectangles_image,
            lsd_lines_component_short_rectangles,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # lsd_lines_component_short_rectangles_zeros_image: 5, 16
        lsd_lines_component_short_rectangles_zeros_image = np.zeros_like(image)
        utilities.draw_rectangles(
            lsd_lines_component_short_rectangles_zeros_image,
            lsd_lines_component_short_rectangles,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # Note that rectangle_component_rectangles was already computed during
        # the execution of the actual algorithm. But since the set of rectangles
        # denoted by the variable got filtered, we perform a recalculation
        # here.
        # rectangle_component_rectangles_image: 7, 18
        rectangle_component_rectangles = (
            general_computer_vision_functions
            .compute_connected_component_rectangles(
                image,
                rectangle_component_parameters
            )
        )
        rectangle_component_rectangles_image = cv.cvtColor(
            lsd_lines_component_rectangles_image,
            cv.COLOR_GRAY2BGR
        )
        utilities.draw_rectangles(
            rectangle_component_rectangles_image,
            rectangle_component_rectangles,
            LSD_LINES_FULL_IMAGE_COLOR,
            RECTANGLE_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # rectangle_component_rectangles_zeros_image: 8, 19
        rectangle_component_rectangles_zeros_image = np.zeros_like(image)
        utilities.draw_rectangles(
            rectangle_component_rectangles_zeros_image,
            rectangle_component_rectangles,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # We perform the same filtering of rectangles as we did during the
        # execution of the actual algorithm, but this time we use an explicitly
        # different variable name.
        # rectangle_component_filtered_rectangles_zeros_image: 9, 20
        if detect_horizontal_lines:
            rectangle_component_filtered_rectangles = (
                geometric_operations.filter_rectangles(
                    rectangle_component_rectangles,
                    horizontal_length_lower_bound=rectangle_length_lower_bound
                )
            )
        else:
            rectangle_component_filtered_rectangles = (
                geometric_operations.filter_rectangles(
                    rectangle_component_rectangles,
                    vertical_length_lower_bound=rectangle_length_lower_bound
                )
            )
        rectangle_component_filtered_rectangles_zeros_image = (
            np.zeros_like(image)
        )
        utilities.draw_rectangles(
            rectangle_component_filtered_rectangles_zeros_image,
            rectangle_component_filtered_rectangles,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # rectangle_component_filtered_rectangles_image: 10, 21
        rectangle_component_filtered_rectangles_image = lsd_lines_image.copy()
        utilities.draw_rectangles(
            rectangle_component_filtered_rectangles_image,
            rectangle_component_filtered_rectangles,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_COLOR,
            LSD_LINES_COMPONENT_RECTANGLES_IMAGE_THICKNESS
        )
        # horizontal_or_vertical_table_lines_rectangles_image: 11, 22
        horizontal_or_vertical_table_lines_rectangles_image = cv.cvtColor(
            rectangle_component_filtered_rectangles_image,
            cv.COLOR_GRAY2BGR
        )
        utilities.draw_lines(
            horizontal_or_vertical_table_lines_rectangles_image,
            horizontal_or_vertical_table_lines,
            TABLE_LINE_COLOR,
            TABLE_LINE_THICKNESS
        )
        # horizontal_or_vertical_table_lines_full_image: 12, 23
        horizontal_or_vertical_table_lines_full_image = (
            cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        )
        utilities.draw_lines(
            horizontal_or_vertical_table_lines_full_image,
            horizontal_or_vertical_table_lines,
            TABLE_LINE_COLOR,
            TABLE_LINE_THICKNESS
        )
        # Note that the progress images lsd_lines_image (examples 3 and 14) and
        # lsd_lines_component_rectangles_image (examples 6 and 17) were
        # constructed during the execution of the actual algorithm.
        progress_images = [
            image,
            lsd_lines_full_image,
            lsd_lines_image,
            lsd_lines_component_short_rectangles_image,
            lsd_lines_component_short_rectangles_zeros_image,
            lsd_lines_component_rectangles_image,
            rectangle_component_rectangles_image,
            rectangle_component_rectangles_zeros_image,
            rectangle_component_filtered_rectangles_zeros_image,
            rectangle_component_filtered_rectangles_image,
            horizontal_or_vertical_table_lines_rectangles_image,
            horizontal_or_vertical_table_lines_full_image
        ]
    else:
        progress_images = None
    return horizontal_or_vertical_table_lines, progress_images

# The function below is used to detect the table structure in the input image,
# i.e., the relevant table lines. This function can also be used to construct
# so-called progress images (images which illustrate the functioning of the
# table line detection algorithm) and/or an image which displays the detected
# table lines.

# To start with, the line-like structures in the input image are detected by
# the detect method of the instance lsd_line_detector of the LSDDetector class.

# The arguments of the detect method are rather technical in nature and will
# not be discussed in detail here.

# The return value of the detect method is a collection of objects called
# lsd_lines. In terms of geometry, an lsd_line is a line segment, but as an
# object data structure it is much more complex. See lsd_line_functions.py for
# functions pertaining to lsd_lines.

# After detecting the relevant line segments (lsd_lines), we call twice the
# function detect_horizontal_or_vertical_table_lines in order to detect
# horizontal and vertical table lines.

# We have the following function argument pairs:
# [horizontal_line_length_lower_bound, vertical_line_length_lower_bound],
# [sin_upper_bound, cos_upper_bound], [right_extra_length, bottom_extra_length],
# [horizontal_rectangle_length_lower_bound,
#  vertical_rectangle_length_lower_bound].

# The first members of the above pairs are used in the detection of horizontal
# table lines and the second members in the detection of vertical table lines.

def detect_table_structure(image,
                           num_octaves,
                           horizontal_line_length_lower_bound,
                           vertical_line_length_lower_bound,
                           sin_upper_bound,
                           cos_upper_bound,
                           right_extra_length,
                           bottom_extra_length,
                           horizontal_rectangle_length_lower_bound,
                           vertical_rectangle_length_lower_bound,
                           construct_progress_images=False,
                           construct_table_line_image=False):
    mask = np.ones_like(image)
    lsd_line_detector = cv.line_descriptor.LSDDetector.createLSDDetector()
    lsd_lines = lsd_line_detector.detect(image,
                                         LSD_LINE_DETECTOR_SCALE,
                                         num_octaves,
                                         mask)
    horizontal_table_lines, horizontal_progress_images = (
        detect_horizontal_or_vertical_table_lines(
            image,
            lsd_lines,
            detect_horizontal_lines=True,
            line_length_lower_bound=horizontal_line_length_lower_bound,
            sin_upper_bound=sin_upper_bound,
            right_extra_length=right_extra_length,
            rectangle_length_lower_bound
            =horizontal_rectangle_length_lower_bound,
            construct_progress_images=construct_progress_images
        )
    )
    vertical_table_lines, vertical_progress_images = (
        detect_horizontal_or_vertical_table_lines(
            image,
            lsd_lines,
            detect_horizontal_lines=False,
            line_length_lower_bound=vertical_line_length_lower_bound,
            cos_upper_bound=cos_upper_bound,
            bottom_extra_length=bottom_extra_length,
            rectangle_length_lower_bound=vertical_rectangle_length_lower_bound,
            construct_progress_images=construct_progress_images
        )
    )
    table_lines = horizontal_table_lines + vertical_table_lines
    if construct_table_line_image:
        table_lines_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        utilities.draw_lines(table_lines_image,
                             table_lines,
                             TABLE_LINE_COLOR,
                             TABLE_LINE_THICKNESS)
    else:
        table_lines_image = None
    table_line_lists = [horizontal_table_lines,
                        vertical_table_lines,
                        table_lines]
    progress_images = [horizontal_progress_images,
                       vertical_progress_images,
                       table_lines_image]
    return table_line_lists, progress_images

# The following function is used to detect the table elements in the input
# image, and also to construct images pertaining to this procedure if needed. 

# A table element is a geometric object which typically represents a local
# collection of shapes, e.g. a printed or handwritten word, a piece of a larger
# drawing, or a stain. The construction of a table element is based on the
# contour detection algorithm provided by cv2. For more details, see the code
# itself.

# Some images illustrating the workings of this function can be found in the
# directory ./example_images/example_result_images. We will use the expression
# table_elements to refer to the file image_3_table_elements.jpg, and we will
# do similarly in the case of other files.

def detect_table_elements(image,
                          table_lines,
                          removed_line_thickness,
                          contour_thickness,
                          construct_table_element_images=False):
    # 1) The input image is binarized by using the Otsu method. A great
    # advantage of the Otsu method is that it does not need user-provided
    # parameters.
    otsu_image = (
        general_computer_vision_functions
        .triangle_or_otsu_binarization(image, otsu_mode=True)
    )
    # 2) Remove the table lines determined earlier from the Otsu image by
    # drawing the table lines in the black color. It is essential that the
    # thickness of the removed lines is chosen to be large enough: The table
    # line detection algorithm is not perfect, so increasing the thickness of
    # the removed lines is needed in order to guarantee that the line-like
    # pixels are removed. On the other hand, if the thickness is too large,
    # significant amounts of other pixels will be removed too. In our tests, we
    # have used the value 20 (see run_main_tests.py).
    # We use the variable black_color in order to make it clear to the reader
    # what the meaning of this particular argument is.
    black_color = 0
    utilities.draw_lines(otsu_image,
                         table_lines,
                         black_color,
                         removed_line_thickness)
    # 3) Detect all contours in the Otsu image. The function detect_contours
    # returns also the so-called hierarchy of contours (not used in the code at
    # the moment), and this is the reason for the [0] index experession.
    contours = (
        general_computer_vision_functions
        .detect_contours(otsu_image)[0]
    )
    # 4) Draw the detected contours in a zero-initialized image of the same
    # shape as the input image. Once again, we use a relatively high value for
    # the thickness (value 20, see run_main_tests.py). The idea is that contours
    # which are close to one another are associated with the same semantic
    # object (e.g. a word or even a paragraph of text), and so such mutually-
    # close contours, when drawn with a large enough thickness, form a blob
    # (more specifically, a connected component) which covers the semantic
    # object mentioned earlier.
    # We use the variable white_color in order to make it clear to the reader
    # what the meaning of this particular argument is.
    blob_image = np.zeros_like(image)
    white_color = 255
    utilities.draw_contours(blob_image,
                            contours,
                            white_color,
                            contour_thickness)
    # 5) Determine the connected components in the image constructed in 4).
    # The resulting data structure table_element_component_parameters represents
    # the information in the input image which does not pertain to table lines.
    table_element_component_parameters = (
        general_computer_vision_functions
        .compute_connected_component_parameters(blob_image)
    )
    # The result image examples table_elements, element_blob_rectangles and
    # element_blobs illustrate the results of this function. The first two of
    # these images are constructed below and element_blobs is constructed by a
    # testing function (see main_test_functions.py).
    if construct_table_element_images:
        full_table_element_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        blob_table_element_image = cv.cvtColor(blob_image, cv.COLOR_GRAY2BGR)
        table_element_rectangles = (
            general_computer_vision_functions
            .compute_connected_component_rectangles(
                image,
                table_element_component_parameters
            )
        )
        utilities.draw_rectangles(full_table_element_image,
                                  table_element_rectangles,
                                  TABLE_ELEMENT_RECTANGLE_COLOR,
                                  TABLE_ELEMENT_RECTANGLE_THICKNESS)
        utilities.draw_rectangles(blob_table_element_image,
                                  table_element_rectangles,
                                  TABLE_ELEMENT_RECTANGLE_COLOR,
                                  TABLE_ELEMENT_RECTANGLE_THICKNESS)
        table_element_images = [full_table_element_image,
                                blob_table_element_image]
    else:
        table_element_images = None
    return table_element_component_parameters, table_element_images

# The following function is the main function in this file, and it is called by
# the functions used to test the main algorithm.

# There are two primary subtasks, namely the detection of the table structure,
# i.e., the relevant table lines, and the detection of table elements (basically
# everything meaningful except for the relevant table lines).

# In addition to the input image, the function receives a set of arguments
# pertaining to each of the primary subtasks.

# The structure of the function itself is hopefully rather self-explanatory due
# to the explicit argument and function names. For a more detailed description
# of the nature of the arguments and return values, see the other functions in
# this file.

def detect_table_structure_and_elements(image,
                                        table_structure_detection_arguments,
                                        table_element_detection_arguments):
    table_line_lists, progress_images = (
        detect_table_structure(
            image,
            *table_structure_detection_arguments
        )
    )
    table_lines = table_line_lists[2]
    table_element_component_parameters, table_element_images = (
        detect_table_elements(
            image,
            table_lines,
            *table_element_detection_arguments
        )
    )
    table_structure_and_elements_description = [
        table_line_lists,
        progress_images,
        table_element_component_parameters,
        table_element_images
    ]
    return table_structure_and_elements_description
