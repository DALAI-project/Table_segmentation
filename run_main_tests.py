# Vesa Ala-Mattila
# Alpha Logos Software Oy
# 31.8.2023
# Modified by Mikko Lipsanen (6.9.2023)

import argparse
import main_test_functions

parser = argparse.ArgumentParser('Arguments for running table segmentation functions.')

parser.add_argument('--INPUT_DIR', type=str, default='./data',
                    help='Directory path for input images.')
parser.add_argument('--RESULTS_DIR', type=str, default='./results',
                    help='Directory path for root directory of the results of the segmentation.')
parser.add_argument('--ARRAYS_SAVE_SUBDIR', type=str, default='arrays',
                    help='Directory path for saving the result arrays.')
parser.add_argument('--IMAGES_SAVE_SUBDIR', type=str, default='images',
                    help='Directory path for saving the result images.')
parser.add_argument('--PROGRESS_IMAGES_SAVE_SUBDIR', type=str, default='progress_images',
                    help='Directory path for saving the progress images.')
parser.add_argument('--RUN_RANDOM_SAMPLE_TEST', action='store_true', 
                    help='Argument defining whether random pages of random documents are processed and the results displayed onscreen.')
parser.add_argument('--NUM_OCTAVES', type=int, default=4,
                    help='Technical parameter for the LSDDetector class of OpenCV.')
parser.add_argument('--HORIZONTAL_LINE_LENGTH_LOWER_BOUND', type=int, default=50,
                    help='Argument for horizontal line detection.')
parser.add_argument('--VERTICAL_LINE_LENGTH_LOWER_BOUND', type=int, default=50,
                    help='Argument for vertical line detection.')
parser.add_argument('--SIN_UPPER_BOUND', type=int, default=0.1,
                    help='Argument for line detection.')
parser.add_argument('--COS_UPPER_BOUND', type=int, default=0.1,
                    help='Argument for line detection.')
parser.add_argument('--RIGHT_EXTRA_LENGTH', type=int, default=150,
                    help='Argument for line detection.')
parser.add_argument('--BOTTOM_EXTRA_LENGTH', type=int, default=300,
                    help='Argument for line detection.')
parser.add_argument('--HORIZONTAL_RECTANGLE_LENGTH_LOWER_BOUND', type=int, default=750,
                    help='Argument for line detection.')
parser.add_argument('--VERTICAL_RECTANGLE_LENGTH_LOWER_BOUND', type=int, default=1500,
                    help='Argument for line detection.')
parser.add_argument('--CONSTRUCT_PROGRESS_IMAGES', action='store_false',
                    help='Argument defining whether images illustrating the functioning of the table line detection algorithm are created.')
parser.add_argument('--CONSTRUCT_TABLE_LINE_IMAGE', action='store_false',
                    help='Argument defining whether images showing the detected table lines are created.')
parser.add_argument('--REMOVED_LINE_THICKNESS', type=int, default=20,
                    help='Argument for table element detection.')
parser.add_argument('--CONTOUR_THICKNESS', type=int, default=20,
                    help='Argument for table element detection.')
parser.add_argument('--CONSTRUCT_TABLE_ELEMENT_IMAGES', action='store_false',
                    help='Argument defining whether images showing the detected table elements are created.')
parser.add_argument('--CONSTRUCT_TABLE_ELEMENT_CELL_POSITION_IMAGE', action='store_false', 
                    help='Argument defining whether table element cell position analysis image is created.')

args = parser.parse_args()

# Result array filename related variables.
HORIZONTAL_TABLE_LINES_FILE_SUFFIX = 'horizontal_table_lines'
VERTICAL_TABLE_LINES_FILE_SUFFIX = 'vertical_table_lines'
TABLE_LINES_FILE_SUFFIX = 'table_lines'
COMPRESSED_ELEMENT_LABEL_ARRAY_FILE_SUFFIX = (
    'compressed_element_label_array'
)
ELEMENT_RECTANGLE_ARRAY_FILE_SUFFIX = 'element_rectangle_array'
ELEMENT_CENTROID_ARRAY_FILE_SUFFIX = 'element_centroid_array'

# Result image filename related variables.
ORIGINAL_IMAGE_FILE_SUFFIX = 'original'
TABLE_LINE_IMAGE_FILE_SUFFIX = 'table_lines'
FULL_TABLE_ELEMENT_IMAGE_FILE_SUFFIX = 'table_elements'
BLOB_TABLE_ELEMENT_IMAGE_FILE_SUFFIX = 'element_blob_rectangles'
ELEMENT_LABEL_IMAGE_FILE_SUFFIX = 'element_blobs'
TABLE_ELEMENT_CELL_POSITION_IMAGE_FILE_SUFFIX = 'table_element_cell_positions'

# Collect the global variables into appropriate lists. These will be used as
# collective function arguments which are unpacked when needed.
table_structure_detection_arguments = [
    args.NUM_OCTAVES,
    args.HORIZONTAL_LINE_LENGTH_LOWER_BOUND,
    args.VERTICAL_LINE_LENGTH_LOWER_BOUND,
    args.SIN_UPPER_BOUND,
    args.COS_UPPER_BOUND,
    args.RIGHT_EXTRA_LENGTH,
    args.BOTTOM_EXTRA_LENGTH,
    args.HORIZONTAL_RECTANGLE_LENGTH_LOWER_BOUND,
    args.VERTICAL_RECTANGLE_LENGTH_LOWER_BOUND,
    args.CONSTRUCT_PROGRESS_IMAGES,
    args.CONSTRUCT_TABLE_LINE_IMAGE
]
table_element_detection_arguments = [
    args.REMOVED_LINE_THICKNESS,
    args.CONTOUR_THICKNESS,
    args.CONSTRUCT_TABLE_ELEMENT_IMAGES
]
data_dir = args.INPUT_DIR

save_dirs_to_create = [
    args.RESULTS_DIR,
    args.ARRAYS_SAVE_SUBDIR,
    args.IMAGES_SAVE_SUBDIR,
    args.PROGRESS_IMAGES_SAVE_SUBDIR
]
result_array_file_suffixes = [
    HORIZONTAL_TABLE_LINES_FILE_SUFFIX,
    VERTICAL_TABLE_LINES_FILE_SUFFIX,
    TABLE_LINES_FILE_SUFFIX,
    COMPRESSED_ELEMENT_LABEL_ARRAY_FILE_SUFFIX,
    ELEMENT_RECTANGLE_ARRAY_FILE_SUFFIX,
    ELEMENT_CENTROID_ARRAY_FILE_SUFFIX
]
result_image_file_suffixes = [
    ORIGINAL_IMAGE_FILE_SUFFIX,
    TABLE_LINE_IMAGE_FILE_SUFFIX,
    FULL_TABLE_ELEMENT_IMAGE_FILE_SUFFIX,
    BLOB_TABLE_ELEMENT_IMAGE_FILE_SUFFIX,
    ELEMENT_LABEL_IMAGE_FILE_SUFFIX,
    TABLE_ELEMENT_CELL_POSITION_IMAGE_FILE_SUFFIX
]

if args.RUN_RANDOM_SAMPLE_TEST:
    main_test_functions.random_sample_test(
        table_structure_detection_arguments,
        table_element_detection_arguments,
        args.CONSTRUCT_TABLE_ELEMENT_CELL_POSITION_IMAGE,
        data_dir
    )
else:
    main_test_functions.multiple_logbooks_test(
        table_structure_detection_arguments,
        table_element_detection_arguments,
        args.CONSTRUCT_TABLE_ELEMENT_CELL_POSITION_IMAGE,
        data_dir,
        save_dirs_to_create,
        result_array_file_suffixes,
        result_image_file_suffixes
    )
