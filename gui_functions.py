
# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np
import cv2 as cv

# The global variable KEY_CODE_DICT maps letters to keyboard command codes used
# by cv2.

# The keyboard commands relevant to this file's functions are the following:

# e: toggle between the overview mode and the zoom mode
# z: cycle through the list of images
# w: move the viewing window up in the zoom mode
# a: move the viewing window left in the zoom mode
# s: move the viewing window down in the zoom mode
# d: move the viewing window right in the zoom mode
# q: stop viewing the images

# There are many more codes included in KEY_CODE_DICT than are used by the
# current version of the code. This is because we used the functions in this
# file when calibrating interactively a number of computer vision algorithms.
# These calibration functions are not included in the current version of the
# code. Since these or similar calibration functions may be needed in the
# future, we include the more extensive keyboard command dictionary.

KEY_CODE_DICT = {'a': 97,
                 'b': 98,
                 'd': 100,
                 'e': 101,
                 'f': 102,
                 'g': 103,
                 'h': 104,
                 'i': 105,
                 'j': 106,
                 'l': 108,
                 'm': 109,
                 'q': 113,
                 'r': 114,
                 's': 115,
                 't': 116,
                 'u': 117,
                 'v': 118,
                 'w': 119,
                 'x': 120,
                 'y': 121,
                 'z': 122}

# The rest of the global variables pertain to the overview and zoom modes.

# FULL_VIEW_SCALE_FACTOR is used to scale down the height and width of the full
# image being viewed when in the overview mode.

# VIEW_WINDOW_SIZE gives the size of the viewing window used while the zoom
# mode is on.

# VIEW_WINDOW_STEP gives the size of the step taken when the viewing window is 
# moved.

# If ZOOM_OUT is True or False, the display mode is the overview mode or the 
# zoom mode, respectively.

# X_MIN and Y_MIN give the current coordinates of the top-left corner point of 
# the viewing window.

FULL_VIEW_SCALE_FACTOR = 0.15
VIEW_WINDOW_SIZE = 750
VIEW_WINDOW_STEP = 700
ZOOM_OUT = True
X_MIN = 0
Y_MIN = 0

# The following function is the secondary function in this file, and it is
# called by the primary function display_multiple_images.

# This function is used to toggle between the overview mode and the zoom mode,
# to move the viewing window around while the zoom mode is on, and to determine
# and display the current image.

# See the code itself for technical details.

def display_image(image, title, key):
    global ZOOM_OUT
    global X_MIN
    global Y_MIN
    height = image.shape[0]
    width = image.shape[1]
    # Toggle between display modes.
    if key == KEY_CODE_DICT['e']:
        ZOOM_OUT = not ZOOM_OUT
    # Move the viewing window.
    if not ZOOM_OUT:
        if key == KEY_CODE_DICT['w']:
            Y_MIN = max(0, Y_MIN - VIEW_WINDOW_STEP)
        if key == KEY_CODE_DICT['a']:
            X_MIN = max(0, X_MIN - VIEW_WINDOW_STEP)
        if key == KEY_CODE_DICT['s']:
            Y_MIN = min(height - VIEW_WINDOW_SIZE, Y_MIN + VIEW_WINDOW_STEP)
        if key == KEY_CODE_DICT['d']:
            X_MIN = min(width - VIEW_WINDOW_SIZE, X_MIN + VIEW_WINDOW_STEP)
    # Determine image_to_display, i.e., the current image to display. It is
    # either a scaled-down version of the full image or the part of the full
    # image determined by the viewing window.
    if ZOOM_OUT:
        scaled_width = int(float(width) * FULL_VIEW_SCALE_FACTOR)
        scaled_height = int(float(height) * FULL_VIEW_SCALE_FACTOR)
        image_to_display = cv.resize(image,
                                     (scaled_width, scaled_height),
                                     interpolation=cv.INTER_LINEAR)
    else:
        x_max = X_MIN + VIEW_WINDOW_SIZE
        y_max = Y_MIN + VIEW_WINDOW_SIZE
        image_to_display = image[Y_MIN:y_max, X_MIN:x_max]
    # Display the current image. The argument title is simply a string giving
    # the title of the image.
    cv.imshow(title, image_to_display)

# The function below is the primary function of this file, and it is the one
# called from outside this file even if the number of images to show is one.

# See the remark *) at the end of this file for a technical comment about using
# cv2 to display images.

# In addition to calling display_image in order to display the current image,
# this function detects keyboard commands (this is done using waitKey) and
# makes it possible for the user to cycle through the relevant list of images
# or to stop viewing the images altogether.

def display_multiple_images(images_to_display, title):
    num_images = len(images_to_display)
    image_index = 0
    done = False
    key = -1
    while not done:
        image = images_to_display[image_index]
        display_image(image, title, key)
        # Detect a keyboard command.
        key = cv.waitKey(0)
        # Cycle through the list of images.
        if key == KEY_CODE_DICT['z']:
            image_index = (image_index + 1) % num_images
        # Stop viewing the images.
        if key == KEY_CODE_DICT['q']:
            cv.destroyAllWindows()
            done = True

# *) The minimal code needed to display an image described by a title consists
# essentially of the lines cv.imshow(title, image) and cv.waitKey(0). The
# command cv.waitKey(0) makes the program wait for a keyboard command for an
# indefinite amount of time. It can be replaced by other commands, see the cv2
# documentation for details on this, but the essential point is that it is not
# sufficient to call imshow alone. To display an image, some GUI housekeeping
# tasks need to be done, and this is not accomplished by calling imshow.
