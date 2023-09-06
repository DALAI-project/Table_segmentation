# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023

import numpy as np

# The following function compresses a numpy array to some extent by extracting
# the non-zero values and the corresponding row and column indices.

# The result is a numpy array with three columns such that the first row
# contains the height and width of the input array, and each of the other
# rows contains the index and value information pertaining to one non-zero
# value of the input array. See the code for the technical details.

# (At the moment, numpy does not have sparse matrices implemented, and we did
# not want to introduce a new library, for example scipy or tensorflow, in
# order to have them, and so we wrote our own rather simple function.)

def construct_compressed_array(array):
    # Find the indices of non-zero values.
    y_indices, x_indices = np.nonzero(array)
    # Collect the non-zero values.
    values = array[y_indices, x_indices]
    # Construct a numpy array whose columns are y_indices, x_indices and values.
    data_type = array.dtype
    indices_and_values_array = np.transpose(np.array(
        [y_indices, x_indices, values],
        data_type
    ))
    # Construct a numpy array that has 3 columns and whose number of rows is
    # one larger than the number of rows of the array constructed above.
    num_nonzero_values = len(y_indices)
    compressed_array = np.empty([num_nonzero_values + 1, 3], data_type)
    # The first row contains the height and width of the input array.
    height, width = array.shape
    compressed_array[0] = np.array([height, width, 0], data_type)
    # The rest of the array contains the indices and non-zero values.
    compressed_array[1:, :] = indices_and_values_array
    return compressed_array

# The function below is simply the inverse of construct_compressed_array.

def construct_array_from_compressed_array(compressed_array):
    # The first row is of the form [height, width, 0].
    height, width, _ = compressed_array[0].astype(np.int32)
    data_type = compressed_array.dtype
    # Initialize the output array with zeros.
    array = np.zeros([height, width], data_type)
    # Unpack the information pertaining to non-zero values and their indices.
    indices_and_values_array = compressed_array[1:, :].transpose()
    y_indices = indices_and_values_array[0]
    x_indices = indices_and_values_array[1]
    values = indices_and_values_array[2]
    # Set the non-zero values.
    array[y_indices, x_indices] = values
    return array

# The general computer vision function compute_connected_component_parameters
# returns a list of four objects such that the second object is a label array
# containing information about the connected components of the input image.

# More specifically, the label array has as values the labels 0, 1, 2, ...,
# N - 1, where 0 corresponds to the background and the other labels to the
# connected components of the input image, so that the label array gives for
# each pixel the connected component containing the pixel.

# The purpose of the following function is to construct an image corresponding
# to the label array, see image_5_element_blobs.jpg in the directory
# ./example_images/example_result_images.

# The current code of the function is simple and probably relatively fast, but
# the downside is that any component whose label is a multiple of 256 will be
# assigned the background color (i.e., black), because the construction is
# based on the mod 256 operation. However, since the downside rarely affects
# more than two components in a given input image, we regard it as a problem
# that can be ignored for the time being.

def construct_label_image(label_array):
    # Construct the color channels by applying the mod 256 operation three
    # times to potentially scaled copies of label_array.
    blue_values = np.mod(label_array, 256).astype(np.uint8)
    green_values = np.mod(label_array * 9, 256).astype(np.uint8)
    red_values = np.mod(label_array * 99, 256).astype(np.uint8)
    # Stack the color channels together in order to form a BGR image.
    label_image = np.dstack((blue_values, green_values, red_values))
    return label_image

# The function below is a slower version of the function construct_label_image
# which does not have the downside mentioned above.

# This version is not used by the current version of the code, so we will not
# discuss it in great detail.

def construct_label_image_slow_version(label_array):
    # Extract the label range.
    min_label = label_array.min()
    max_label = label_array.max()
    assert min_label == 0
    # Initialize label_image with zeros.
    height, width = label_array.shape
    label_image = np.zeros([height, width, 3], np.uint8)
    # The label 0 corresponds to the background, so the first label of a real
    # connected component is 1.
    for label in range(1, max_label + 1):
        # Choose a random color.
        blue_value = np.random.randint(0, 256)
        green_value = np.random.randint(0, 256)
        red_value = np.random.randint(0, 256)
        color = [blue_value, green_value, red_value]
        # Determine the indices of the pixels corresponding to the current
        # label.
        label_y_indices, label_x_indices = np.where(label_array == label)
        # Color the relevant pixels appropriately.
        label_image[label_y_indices, label_x_indices] = color
    return label_image
