
# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

import general_computer_vision_functions
import utilities

# The following global variables are used in the construction of images which
# illustrate the results of the main function of this file, so there is no
# need to elaborate on them further.

TABLE_LINES_COLOR = (255, 0, 0)
TABLE_LINES_THICKNESS = 5
TABLE_ELEMENT_RECTANGLE_THICKNESS = 2

# The following function computes a sorted collection of mean coordinates for a
# given set of horizontal/vertical table lines.

# If the lines are horizontal, the coordinates are y-coordinates, and the
# coordinates are x-coordinates in case the lines are vertical.

# More specifically, for a horizontal table line, the computed coordinate is
# the y-coordinate of the middle point of the table line, and for a vertical
# table line, we get the x-coordinate of the middle point.

# Basically, this function is used to compute the positions of the lines in
# the xy-grid that approximates the actual table.

# The computed coordinates are sorted, since this is important for future
# steps of the algorithm.

# Additionally, if 0 is not in the resulting collection of coordinates, it is
# added into the collection. This is done in order to guarantee that a given
# table element will always have a grid line on its left side and another one
# above it.

def compute_sorted_mean_coordinates(lines, lines_are_horizontal):
    # The means are computed with respect to the y or x direction depending on
    # whether the given collection of lines is horizontal or vertical,
    # respectively.
    # We use the set construction in order to remove duplicates.
    if lines_are_horizontal:
        sorted_mean_coordinates = sorted(set([
            int((y_1 + y_2) / 2) for (_, y_1), (_, y_2) in lines
        ]))
    else:
        sorted_mean_coordinates = sorted(set([
            int((x_1 + x_2) / 2) for (x_1, _), (x_2, _) in lines
        ]))
    # Add 0 if it is missing from the collection. This guarantees that a given
    # table element will always have a grid line on its left side and another
    # one above it.
    if 0 not in sorted_mean_coordinates:
        sorted_mean_coordinates = [0] + sorted_mean_coordinates
    # The sorted list is made into a numpy array in anticipation of later steps
    # of the algorithm.
    sorted_mean_coordinates = np.array(sorted_mean_coordinates)
    return sorted_mean_coordinates

# Assuming that a given table element is in a given grid cell, one can use the
# following function to determine the grid lines of the cell (called cell
# lines).

# More specifically, the left cell line and the top cell line are explicitly
# determined using this function.

# Let us consider the case of the left cell line only, since the case of the
# top cell line is similar.

# In this case, the given collection of sorted mean coordinates contains the
# x-positions of the vertical grid lines.

# The coordinate argument is the x-coordinate of the center point p of the
# minimal rectangle containing the given table element.

# We first compute the signed distances of p from the vertical grid lines.

# Let d_min be the signed distance with the minimal absolute value, and let
# L be the grid line closest to p (so the signed distance between p and L is
# d_min).

# If d_min is negative, it means that L is on the left-hand side of p and,
# therefore, L is the left cell line we are looking for.

# If d_min is positive, then L is on the right-hand side of p, and so the left
# cell line we want is the grid line on the left-hand side of L.

def determine_cell_line(coordinate, sorted_mean_coordinates):
    # Determine the closest grid line by computing the signed distances and
    # choosing the one with the minimal absolute value.
    signed_distances = sorted_mean_coordinates - coordinate
    minimal_distance_index = np.min(np.argmin(np.abs(signed_distances)))
    minimal_signed_distance = signed_distances[minimal_distance_index]
    # If the minimal signed distance is negative, the closest grid line is the
    # left/top cell line, and this is the cell line we want to find.
    if minimal_signed_distance <= 0:
        cell_line_index = minimal_distance_index
    # If the minimal signed distance is positive, the closest grid line is the
    # right/bottom cell line, and so in order to get the left/top cell line,
    # we need to move one grid step to the left/up.
    else:
        cell_line_index = minimal_distance_index - 1
    return cell_line_index

# The function below determines the appropriate grid cell for each table
# element. In technical terms, the following takes place.

# Let p be the center point of the minimal rectangle containing the given
# table element.

# We use determine_cell_line to determine the top and left cell lines of p.

# The top and left cell lines determine a unique cell, and so we can place
# p in this cell and, therefore, also the element itself and its minimal
# rectangle.

# In the code below, we associate the minimal rectangle with the cell because
# this facilitates the construction of an image that describes the results of
# the algorithm.

def determine_table_element_cell_positions(table_element_rectangles,
                                           y_means,
                                           x_means):
    # The cell positions are represented by a dictionary whose keys are of the
    # form (top_cell_line_index, left_cell_line_index) and whose values are
    # lists containing minimal rectangles of table elements.
    table_element_cell_positions = {}
    for rectangle in table_element_rectangles:
        (x_1, y_1), (x_2, y_2) = rectangle
        # Determine the center point of the rectangle.
        x_center = int((x_1 + x_2) / 2)
        y_center = int((y_1 + y_2) / 2)
        # Determine the top and left cell lines of the center point. These two
        # cell lines determine the identity of the cell.
        top_cell_line_index = determine_cell_line(y_center, y_means)
        left_cell_line_index = determine_cell_line(x_center, x_means)
        cell = (top_cell_line_index, left_cell_line_index)
        # Add the rectangle to the list corresponding to the cell.
        non_empty_cells = table_element_cell_positions.keys()
        if cell not in non_empty_cells:
            table_element_cell_positions[cell] = [rectangle]
        else:
            table_element_cell_positions[cell].append(rectangle)
    return table_element_cell_positions

# The following simple function is used to construct images which illustrate
# the results of the main function.

def construct_table_element_cell_position_image(image,
                                                table_lines,
                                                table_element_cell_positions):
    table_element_cell_position_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # Draw first the table lines.
    utilities.draw_lines(
        table_element_cell_position_image,
        table_lines,
        TABLE_LINES_COLOR,
        TABLE_LINES_THICKNESS
    )
    # Next, draw the table element rectangles so that the rectangles associated
    # with a particular cell are all drawn using the same color. The color
    # corresponding to a cell is chosen randomly.
    for cell, rectangles in table_element_cell_positions.items():
        blue_value = np.random.randint(0, 256)
        green_value = np.random.randint(0, 256)
        red_value = np.random.randint(0, 256)
        color = [blue_value, green_value, red_value]
        utilities.draw_rectangles(
            table_element_cell_position_image,
            rectangles,
            color,
            TABLE_ELEMENT_RECTANGLE_THICKNESS
        )
    return table_element_cell_position_image


def table_element_position_analysis(image,
                                    table_structure_and_elements_description):
    # Do some argument unpacking.
    table_line_lists = table_structure_and_elements_description[0]
    table_element_component_parameters = (
        table_structure_and_elements_description[2]
    )
    horizontal_table_lines = table_line_lists[0]
    vertical_table_lines = table_line_lists[1]
    table_lines = table_line_lists[2]
    # Compute the minimal table element rectangles.
    table_element_rectangles = (
        general_computer_vision_functions
        .compute_connected_component_rectangles(
            image,
            table_element_component_parameters
        )
    )
    # Compute the positions of the lines in the xy-grid which approximates the
    # actual table.
    y_means = compute_sorted_mean_coordinates(horizontal_table_lines,
                                              lines_are_horizontal=True)
    x_means = compute_sorted_mean_coordinates(vertical_table_lines,
                                              lines_are_horizontal=False)
    # Place the table elements into grid cells.
    table_element_cell_positions = (
        determine_table_element_cell_positions(
            table_element_rectangles,
            y_means,
            x_means
        )
    )
    # Draw an image describing the results.
    table_element_cell_position_image = (
        construct_table_element_cell_position_image(
            image,
            table_lines,
            table_element_cell_positions
        )
    )
    return table_element_cell_position_image
