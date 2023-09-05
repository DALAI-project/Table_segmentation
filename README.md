# Table segmentation

<img src="table_example.jpg"  width="40%" height="40%">

The code, provided by [Alpha Logos Software Oy](https://www.alphalogos.fi/), contains several functions used for detecting table lines as well as content elements from tables in digitized document images. The documentation of the code is provided in a separate `documentation.pdf` file, and more detailed information on many of the functions and parameters can be found from the documentation of the [OpenCV](https://opencv.org/) library. The code has been developed specifically to detect table lines and content elements from Finnish ship logbooks from the late 19th - early 20th centuries (see example image above), and the default parameter values are optimized for that dataset. 

## Installation

- Create and activate conda environment:

`conda create -n table_env python=3.11`

`conda activate table_env`

- Install required libraries:

`pip install -r requirements.txt`

## Running the code

Image segmentation can be performed by running the `run_segment.py` file using the command line. There are a variety of parameters that can be provided as input to the code, some of which are listed below. 

By default, the code expects input images to be located in subfolders of the `/data` folder, and the results are placed in subfolders of the `/results` folder. When the default folder names are used and the output is chosen to include table line images, progress images, table element images and table element cell position images, the following folder structure is expected before running the code:

```
├──Table_segmentation
      ├──results 
      ├──data
      |   ├──image_folder_1
      |   |   ├──img_1_1.jpg...
      |   └──image_folder_2
      |       ├──img_2_1.jpg...
      ├──run_segment.py
      ├──main_functions.py
      ├──utilities.py
      ├──requirements.txt
      ...
```
After running the code, the `/results` folder content should have the following subfolders, which contain the result files:

```
├──Table_segmentation
      ├──results 
      |   ├──image_folder_1
      |   |   ├──arrays
      |   |   ├──images
      |   |   ├──progress_images
      |   |   └──numbers_of_table_elements.npy
      |   └──image_folder_2
      |       ...
      ...
```
## Parameters

- `INPUT_DIR` defines the folder where the input image data is located. The image files are expected to be located in subfolders of the `INPUT_DIR`, which can be named freely by the user. Default folder path for the input directory is `./data`.
- `RESULTS_DIR` defines the folder where the results of the functions are saved. The number and types of subfolders depends on the user's choice of outputs. Default results folder path is `./results`.
- 

