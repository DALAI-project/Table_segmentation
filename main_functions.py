# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023
# Modified by Mikko Lipsanen (6.9.2023)

import numpy as np
import cv2 as cv
import time
import os

import main_computer_vision_functions
import numpy_array_operations
import analysis_functions
import gui_functions
import utilities

# The following simple function constructs a list containing all of the
# data to be processed by multiple_document_test.

# The function is essentially trivial, but the reasoning for its existence
# follows from our decision not to do any collective function argument
# unpacking in the main test functions themselves.


def construct_document_list(root_dir):
    document_list = os.listdir(root_dir)
    return document_list

# The function below loads the input images associated with a given document.
def load_page_images(logbook, root_dir):
    image_dir = os.path.join(root_dir, logbook)
    images = utilities.load_images(image_dir, grayscale=True)
    return images

# The function random_sample_test uses the following function to load the
# images it processes one at a time.

def load_random_logbook_page_image(data_dir):
    logbooks_root_dir = data_dir
    logbook_list = os.listdir(logbooks_root_dir)
    logbook = np.random.choice(logbook_list)
    image_dir = os.path.join(logbooks_root_dir, logbook)
    image_files = os.listdir(image_dir)
    image_file = np.random.choice(image_files)
    image_path = os.path.join(image_dir, image_file)
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, logbook, image_file

# For a given logbook, the following function creates the required save
# directories.

def create_save_directories(logbook,
                            save_dirs_to_create,
                            table_structure_detection_arguments):
    # Unpack the arguments.
    save_root_dir = save_dirs_to_create[0]
    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)
    subdirs_to_create = save_dirs_to_create[1:]
    construct_progress_images = table_structure_detection_arguments[9]
    # We need to create logbook_save_dir and logbook_save_dir/subdir for every
    # subdir in subdirs_to_create. 
    logbook_save_dir = os.path.join(save_root_dir, logbook)
    # By adding the empty string to subdirs_to_create, we can use the
    # following loop to create logbook_save_dir as well.
    subdirs_to_create = [''] + subdirs_to_create
    # Remove the last subdirectory if it is not needed.
    if not construct_progress_images:
        subdirs_to_create = subdirs_to_create[:-1]
    # Create all of the required directories by using the following loop.
    save_dirs = []
    for subdir in subdirs_to_create:
        # If subdir == '', then directory_to_create == logbook_save_dir.
        directory_to_create = os.path.join(logbook_save_dir, subdir)
        # Create the directory if it does not already exist.
        if not os.path.isdir(directory_to_create):
            os.mkdir(directory_to_create)
        save_dirs.append(directory_to_create)
    return save_dirs

# The following is a straightforward function for saving result arrays.

def save_result_arrays(result_arrays,
                       image_number,
                       save_dirs,
                       file_suffixes):
    arrays_save_dir = save_dirs[1]
    for result_array, file_suffix in zip(result_arrays, file_suffixes):
        filename = 'image_{}_{}.npy'.format(image_number, file_suffix)
        path = os.path.join(arrays_save_dir, filename)
        np.save(path, result_array)

# The function below is used to save result images.

# The list result_images has already been prepared by prepare_result_images
# (the code of this function can be found below), so only the corresponding
# list of file suffixes has to be constructed, which is a simple task.

# The variable result_image_number is used in the variable filename in order to
# guarantee that the resulting result image files have a certain order.

def save_result_images(result_images,
                       image_number,
                       save_dirs,
                       table_structure_detection_arguments,
                       table_element_detection_arguments,
                       construct_table_element_cell_position_image,
                       all_file_suffixes):
    # Unpack the relevant Boolean arguments.
    construct_table_line_image = table_structure_detection_arguments[10]
    construct_table_element_images = table_element_detection_arguments[2]
    # Construct the appropriate list of file suffixes.
    file_suffixes = [all_file_suffixes[0]]
    if construct_table_line_image:
        file_suffixes.append(all_file_suffixes[1])
    if construct_table_element_images:
        file_suffixes.extend(all_file_suffixes[2:5])
    if construct_table_element_cell_position_image:
        file_suffixes.append(all_file_suffixes[5])
    # Save the images.
    images_save_dir = save_dirs[2]
    image_suffix_pairs = zip(result_images, file_suffixes) 
    for p, (result_image, file_suffix) in enumerate(image_suffix_pairs):
        # The addition of result_image_number into filename guarantees that
        # the result image files have a certain appropriate order.
        result_image_number = p + 1
        filename = 'image_{}_{}_{}.jpg'.format(image_number,
                                               result_image_number,
                                               file_suffix)
        path = os.path.join(images_save_dir, filename)
        cv.imwrite(path, result_image)

# This function saves progress images related to the detection of table lines,
# see ./example_images/example_progress_images.

# The reason we make this function return a time variable pertaining to the
# function operation is that we want to avoid unpacking collective function
# arguments in the main test functions.

def save_progress_images(image_number,
                         table_structure_and_elements_description,
                         table_structure_detection_arguments,
                         save_dirs):
    construct_progress_images = table_structure_detection_arguments[9]
    if construct_progress_images:
        progress_images = table_structure_and_elements_description[1]
        # The lists of horizontal and vertical progress images both contain the 
        # original input image as the first element, so in order to avoid
        # including the original input image twice, we add the [1:] part to the
        # expression featuring vertical_progress_images.
        horizontal_progress_images = progress_images[0]
        vertical_progress_images = progress_images[1][1:]
        # Strictly speaking, table_line_image is not a progress image but
        # rather a result image. However, it is natural to include it into
        # the collection of progress images.
        table_line_image = progress_images[2]
        progress_images_to_save = (
            horizontal_progress_images
            + vertical_progress_images
            + [table_line_image]
        )
        progress_images_save_dir = save_dirs[3]
        for p, progress_image in enumerate(progress_images_to_save):
            progress_image_number = p + 1
            filename = 'image_{}_{}.jpg'.format(image_number,
                                                progress_image_number)
            path = os.path.join(progress_images_save_dir, filename)
            cv.imwrite(path, progress_image)
        progress_images_saved_time = time.time()
    else:
        progress_images_saved_time = None
    return progress_images_saved_time

# A result array is almost always associated with a particular logbook page.
# The exceptional result array is numbers_of_table_elements which is associated
# with a given logbook in its totality.

# In terms of structure, the array consists of rows which are of the form
# [image_number, number_of_table_elements] (see prepare_result_arrays for more
# details), so this array simply records how many table elements were detected
# in each page.

# The function below saves numbers_of_table_elements. Additionally, the
# function computes the maximum and the mean over the numbers of table
# elements contained in the array and prints out the result. 

def study_and_save_numbers_of_table_elements(numbers_of_table_elements,
                                             logbook,
                                             save_dirs):
    numbers_of_table_elements = np.array(numbers_of_table_elements)
    # The actual numbers of table elements are in the second column. The first
    # column contains image numbers.
    element_number_max = numbers_of_table_elements[:, 1].max()
    element_number_mean = numbers_of_table_elements[:, 1].mean()
    # Create a message and print it.
    element_number_string = (
        'Document {} table element numbers: \n'.format(logbook)
        + 'Maximum: {} \n'.format(element_number_max)
        + 'Mean: {:.2f} \n'.format(element_number_mean)
    )
    print(element_number_string)
    # Save the array.
    logbook_save_dir = save_dirs[0]
    filename = 'numbers_of_table_elements.npy'
    path = os.path.join(logbook_save_dir, filename)
    np.save(path, numbers_of_table_elements)

# The following function prepares the page-specific result arrays for saving
# and extends numbers_of_table_elements appropriately (see the the discussion
# on study_and_save_numbers_of_table_elements for details).

# There are six page-specific result arrays. Three of them pertain to table
# lines: one contains the horizontal, another the vertical, and the third all
# of the table lines.

# (Recall that a table line is represented by a pair of endpoints expressed in
# the coordinate system of the input image.)

# The other three pertain to the table elements of the input image. (Recall
# that the current algorithm identifies table elements with certain connected
# components.)

# There is the label array (which we save in a compressed form) that gives for
# each pixel in the input image the table element containing the pixel.

# There is also an array describing the minimal rectangles containing the table
# elements. Such a rectangle is represented by four numbers: the x and
# y-coordinates of the top-left corner point, and the x and y-lengths of the
# rectangle (i.e., the width and the height).

# Finally, there is an array which stores for each table element the
# coordinates (in the coordinate system of the input image) of the
# centroid of the element.

# Note that the rectangle and centroid arrays are saved in their unprocessed
# forms, i.e., the rectangle description format is not the same as used by,
# e.g., drawing functions, and both rectangles contain on their first rows
# information about the background of the input image. We do this because,
# in the current version of the code, rectangles are processed locally when
# needed, and the centroids are not used at all.

def prepare_result_arrays(table_structure_and_elements_description,
                          numbers_of_table_elements,
                          image_number):
    # Unpack table_structure_and_elements_description.
    table_line_lists = table_structure_and_elements_description[0]
    table_element_component_parameters = (
        table_structure_and_elements_description[2]
    )
    # Unpack table_line_lists. This gives us three of the result arrays.
    horizontal_table_lines = np.array(table_line_lists[0])
    vertical_table_lines = np.array(table_line_lists[1])
    table_lines = np.array(table_line_lists[2])
    # Unpack table_element_component_parameters. This immediately gives us two
    # more result arrays, namely element_rectangle_array and
    # element_centroid_array.
    number_of_element_labels = table_element_component_parameters[0]
    element_label_array = np.array(table_element_component_parameters[1])
    element_rectangle_array = np.array(table_element_component_parameters[2])
    element_centroid_array = np.array(table_element_component_parameters[3])
    # The number of actual table elements is one smaller than the number of
    # element labels because the background is given the 0 label.
    number_of_table_elements = number_of_element_labels - 1
    # The result array numbers_of_table_elements is associated with the whole
    # logbook and not just a particular page of the logbook. This is why this
    # array is accumulated one page at a time.
    numbers_of_table_elements.append([image_number, number_of_table_elements])
    # The last result array to be prepared is a compressed form of the element
    # label array.
    compressed_element_label_array = (
        numpy_array_operations.construct_compressed_array(element_label_array)
    )
    result_arrays = [horizontal_table_lines,
                     vertical_table_lines,
                     table_lines,
                     compressed_element_label_array,
                     element_rectangle_array,
                     element_centroid_array]
    return result_arrays

# The following function prepares result images to be saved, see the directory
# ./example_images/example_result_images for a set of examples.

# A full set of result images contains six images.

# The first one is the original input image (in grayscale format).

# The second image displays the table lines found by the main algorithm.

# The next three are constructed by the table element detection algorithm.

# The first one of these shows the input image with the minimal table element
# rectangles drawn in.

# Next we have a binary image showing the blobs (i.e., connected components)
# corresponding to the table elements, and also the minimal rectangles.

# The last image in this set of three displays the blobs using different
# colors. This image is constructed by the function below by letting
# construct_label_image of numpy_array_operations.py act on element_label_array
# (an array considered in detail in the context of prepare_result_arrays).

# The final result image shows again the input image with the minimal element
# rectangles added. This time the colors of the rectangles vary. The idea is
# that elements associated with the same table cell share rectangle color. This
# image is created by table_element_position_analysis contained in the file
# analysis_functions.py. See that file for more details.

def prepare_result_images(image,
                          table_structure_and_elements_description,
                          table_structure_detection_arguments,
                          table_element_detection_arguments,
                          construct_table_element_cell_position_image):
    # Do some initial unpacking of arguments.
    construct_table_line_image = table_structure_detection_arguments[10]
    construct_table_element_images = table_element_detection_arguments[2]
    progress_images = table_structure_and_elements_description[1]
    table_element_images = table_structure_and_elements_description[3]
    # The original input image will always be a result image. Add other images
    # according to the values of the Boolean variables.
    result_images = [image]
    if construct_table_line_image:
        table_line_image = progress_images[2]
        result_images.append(table_line_image)
    if construct_table_element_images:
        table_element_component_parameters = (
            table_structure_and_elements_description[2]
        )
        element_label_array = np.array(table_element_component_parameters[1])
        element_label_image = (
            numpy_array_operations.construct_label_image(element_label_array)
        )
        result_images.extend(table_element_images)
        result_images.append(element_label_image)
    if construct_table_element_cell_position_image:
        table_element_cell_position_image = (
            analysis_functions.table_element_position_analysis(
                image,
                table_structure_and_elements_description
            )
        )
        result_images.append(table_element_cell_position_image)
    return result_images

# The function below determines the images to display after a random_sample_test
# has been performed. This function is very similar to prepare_result_images,
# so we will not consider it in greater detail.

def determine_images_to_display(image,
                                table_structure_and_elements_description,
                                table_structure_detection_arguments,
                                table_element_detection_arguments,
                                construct_table_element_cell_position_image):
    # Do some initial unpacking of arguments.
    construct_progress_images = table_structure_detection_arguments[9]
    construct_table_line_image = table_structure_detection_arguments[10]
    construct_table_element_images = table_element_detection_arguments[2]
    progress_images = table_structure_and_elements_description[1]
    table_element_images = table_structure_and_elements_description[3]
    # The original input image will always be displayed. Add other images
    # according to the values of the Boolean variables.
    images_to_display = [image]
    if construct_progress_images:
        # The list of horizontal/vertical progress images has the input image
        # as its first element. This explains the [1:] part in the following
        # two expressions.
        horizontal_progress_images = progress_images[0][1:]
        vertical_progress_images = progress_images[1][1:]
        images_to_display.extend(horizontal_progress_images)
        images_to_display.extend(vertical_progress_images)
    if construct_table_line_image:
        table_line_image = progress_images[2]
        images_to_display.append(table_line_image)
    if construct_table_element_images:
        table_element_component_parameters = (
            table_structure_and_elements_description[2]
        )
        element_label_array = np.array(table_element_component_parameters[1])
        element_label_image = (
            numpy_array_operations.construct_label_image(element_label_array)
        )
        images_to_display.extend(table_element_images)
        images_to_display.append(element_label_image)
    if construct_table_element_cell_position_image:
        table_element_cell_position_image = (
            analysis_functions.table_element_position_analysis(
                image,
                table_structure_and_elements_description
            )
        )
        images_to_display.append(table_element_cell_position_image)
    return images_to_display

# The following simple, self-explanatory function prints information pertaining
# to a run of random_sample_test.

def print_random_sample_test_times(logbook, image_file, times):
    image_loading_time = times[1] - times[0]
    lines_and_elements_time  = times[2] - times[1]
    total_time = times[2] - times[0]
    random_sample_test_times_string = (
        'Document: {} \n'.format(logbook)
        + 'Image file: {} \n'.format(image_file)
        + 'Image file loading time: {:.2f}s \n'.format(image_loading_time)
        + 'Table lines and elements: {:.2f}s \n'.format(lines_and_elements_time)
        + 'Total time: {:.2f}s \n'.format(total_time)
    )
    print(random_sample_test_times_string)

# The function below prints information after each time multiple_logbooks_test
# has processed an input image. The construction of the string printed is more
# complex than in the function print_random_sample_test_times but still rather
# simple.

def print_multiple_logbooks_test_times(logbook,
                                       logbook_number,
                                       total_number_of_logbooks,
                                       image_number,
                                       total_number_of_images,
                                       table_structure_and_elements_description,
                                       times,
                                       table_structure_detection_arguments):
    # Do some argument unpacking.
    table_element_component_parameters = (
        table_structure_and_elements_description[2]
    )
    number_of_element_labels = table_element_component_parameters[0]
    # Remember that the 0 label refers to the background, so the number of
    # table elements is one less than the number of element labels.
    number_of_table_elements = number_of_element_labels - 1
    # Construct the part of the main string that will printed in any case.
    lines_and_elements_time = times[1] - times[0]
    arrays_prepared_time = times[2] - times[1]
    arrays_saved_time = times[3] - times[2]
    images_prepared_time = times[4] - times[3]
    images_saved_time = times[5] - times[4]
    multiple_logbooks_test_times_string = (
        'Document: {} ({} / {}) \n'.format(logbook,
                                          logbook_number,
                                          total_number_of_logbooks)
        + 'Image: {} / {} \n'.format(image_number, total_number_of_images)
        + 'Number of elements: {} \n'.format(number_of_table_elements)
        + 'Table lines and elements: {:.2f}s \n'.format(lines_and_elements_time)
        + 'Result arrays prepared: {:.2f}s \n'.format(arrays_prepared_time)
        + 'Result arrays saved: {:.2f}s \n'.format(arrays_saved_time)
        + 'Result images prepared: {:.2f}s \n'.format(images_prepared_time)
        + 'Result images saved: {:.2f}s \n'.format(images_saved_time)
    )
    # Extend the main string if needed.
    construct_progress_images = table_structure_detection_arguments[9]
    if construct_progress_images:
        progress_images_saved_time = times[6] - times[5]
        progress_images_string = (
            'Progress images saved: {:.2f}s \n'.format(
                progress_images_saved_time
            )
        )
        multiple_logbooks_test_times_string += progress_images_string
    # Finalize and print the main string.
    total_time = times[-1] - times[0]
    total_time_string = 'Total time: {:.2f}s \n'.format(total_time)
    multiple_logbooks_test_times_string += total_time_string
    print(multiple_logbooks_test_times_string)

# The following simple function prints the elapsed time after
# multiple_logbooks_test has processed a whole logbook.

def print_logbook_total_time(logbook_start_time):
    logbook_end_time = time.time()
    logbook_total_time = logbook_end_time - logbook_start_time
    logbook_total_time_string = (
        'Document total time: {:.2f}s \n'.format(logbook_total_time)
    )
    print(logbook_total_time_string)

# The following is the first main test function. It processes random pages of
# random documents one at a time and displays the results onscreen. Given our
# discussion on the auxiliary functions above, the code of the function should
# be easy to understand.

def random_sample_test(table_structure_detection_arguments,
                       table_element_detection_arguments,
                       construct_table_element_cell_position_image,
                       data_dir):
    while True:
        start_time = time.time()
        # Choose a random page of a random logbook.
        image, logbook, image_file = load_random_logbook_page_image(data_dir)
        image_loaded_time = time.time()
        # Determine table lines and table elements.
        table_structure_and_elements_description = (
            main_computer_vision_functions
            .detect_table_structure_and_elements(
                image,
                table_structure_detection_arguments,
                table_element_detection_arguments
            )
        )
        table_lines_and_elements_obtained_time = time.time()
        # Print a message pertaining to the test run.
        times = [start_time,
                 image_loaded_time,
                 table_lines_and_elements_obtained_time]
        print_random_sample_test_times(logbook, image_file, times)
        # Display images illustrating the test run.
        images_to_display = determine_images_to_display(
            image,
            table_structure_and_elements_description,
            table_structure_detection_arguments,
            table_element_detection_arguments,
            construct_table_element_cell_position_image
        )
        title = 'Document {} / {}'.format(logbook, image_file)
        gui_functions.display_multiple_images(images_to_display, title)

# The following is the second main test function. It processes all of the
# documents in ./data, or more generally, in the root data
# directory, and saves the results in ./results, or more
# generally, in the root result directory. Given our discussion on the
# auxiliary functions above, the code of the function should be easy to
# understand.

def multiple_logbooks_test(table_structure_detection_arguments,
                           table_element_detection_arguments,
                           construct_table_element_cell_position_image,
                           data_dir,
                           save_dirs_to_create,
                           result_array_file_suffixes,
                           result_image_file_suffixes):
    # Get the list of logbooks and start processing the logbooks one at a time.
    #logbook_list = construct_document_list(data_dirs)
    logbook_list = construct_document_list(data_dir)
    total_number_of_logbooks = len(logbook_list)
    for b, logbook in enumerate(logbook_list):
        logbook_start_time = time.time()
        logbook_number = b + 1
        # Create the save directories.
        save_dirs = create_save_directories(
            logbook,
            save_dirs_to_create,
            table_structure_detection_arguments
        )
        # Load the images.
        images = load_page_images(logbook, data_dir)
        total_number_of_images = len(images)
        # The array numbers_of_table_elements is accumulated one image at a
        # time.
        numbers_of_table_elements = []
        for i, image in enumerate(images):
            start_time = time.time()
            image_number = i + 1
            # Determine table lines and table elements.
            table_structure_and_elements_description = (
                main_computer_vision_functions
                .detect_table_structure_and_elements(
                    image,
                    table_structure_detection_arguments,
                    table_element_detection_arguments
                )
            )
            table_lines_and_elements_obtained_time = time.time()
            # Prepare and save result arrays.
            result_arrays = prepare_result_arrays(
                table_structure_and_elements_description,
                numbers_of_table_elements,
                image_number
            )
            result_arrays_prepared_time = time.time()
            save_result_arrays(result_arrays,
                               image_number,
                               save_dirs,
                               result_array_file_suffixes)
            result_arrays_saved_time = time.time()
            # Prepare and save result images.
            result_images = prepare_result_images(
                image,
                table_structure_and_elements_description,
                table_structure_detection_arguments,
                table_element_detection_arguments,
                construct_table_element_cell_position_image
            )
            result_images_prepared_time = time.time()
            save_result_images(
                result_images,
                image_number,
                save_dirs,
                table_structure_detection_arguments,
                table_element_detection_arguments,
                construct_table_element_cell_position_image,
                result_image_file_suffixes)
            result_images_saved_time = time.time()
            # Save progress images if needed.
            progress_images_saved_time = save_progress_images(
                image_number,
                table_structure_and_elements_description,
                table_structure_detection_arguments,
                save_dirs
            )
            # Print a message pertaining to the processing of the input image.
            times = [start_time,
                     table_lines_and_elements_obtained_time,
                     result_arrays_prepared_time,
                     result_arrays_saved_time,
                     result_images_prepared_time,
                     result_images_saved_time]
            if progress_images_saved_time is not None:
                times.append(progress_images_saved_time)
            print_multiple_logbooks_test_times(
                logbook,
                logbook_number,
                total_number_of_logbooks,
                image_number,
                total_number_of_images,
                table_structure_and_elements_description,
                times,
                table_structure_detection_arguments
            )
        # After all of the pages of a logbook have been processed, save the
        # result array numbers_of_table_elements and print a message as to
        # the elapsed time.
        study_and_save_numbers_of_table_elements(
            numbers_of_table_elements,
            logbook,
            save_dirs
        )
        print_logbook_total_time(logbook_start_time)
