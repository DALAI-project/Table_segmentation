# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv
import os

# The function below is used to load images in the current version of the code.

# A whole directory of images is loaded when this function is called. It is
# essential that the directory contains the image files and nothing else.

# The most important technical functions assume the input images to be
# grayscale images, so the grayscale argument is typically given the value
# True.

def load_images(image_dir, grayscale):
    image_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, file) for file in image_files]
    images = [cv.imread(path) for path in image_paths]
    if grayscale:
        images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    return images

# The following three functions are simple drawing functions.

# The meaning of the argument cv.LINE_AA is that antialiasing is enabled. The
# argument -1 used in draw_contours makes the function draw all of the contours
# in the argument collection contours.

def draw_lines(image, lines, color, thickness):
    for line in lines:
        start_point, end_point = line
        cv.line(image,
                start_point,
                end_point,
                color,
                thickness,
                cv.LINE_AA)

def draw_rectangles(image, rectangles, color, thickness):
    for rectangle in rectangles:
        top_left_point, bottom_right_point = rectangle
        cv.rectangle(image,
                     top_left_point,
                     bottom_right_point,
                     color,
                     thickness,
                     cv.LINE_AA)

def draw_contours(image, contours, color, thickness):
    cv.drawContours(image,
                    contours,
                    -1,
                    color,
                    thickness,
                    cv.LINE_AA)

# Instead of drawing a contour itself, the function below makes it possible to
# draw the minimal rectangle containing the contour.

# Note that this minimal rectangle does not have to be xy-aligned.

# The function minAreaRect constructs the minimal rectangle object. The
# function boxPoints reduces the rectangle object into a collection of corner
# point coordinates, and the function intp turns these coordinates into
# integers.

# The meaning of the argument 0 used in drawContours is that the function draws
# the 0-index object in the list [rectangle], i.e., rectangle itself.

def draw_contour_rectangles(image, contours, color, thickness):
    for contour in contours:
        rectangle = cv.minAreaRect(contour)
        rectangle = cv.boxPoints(rectangle)
        rectangle = np.intp(rectangle)
        cv.drawContours(image,
                        [rectangle],
                        0,
                        color,
                        thickness,
                        cv.LINE_AA)
