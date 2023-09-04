# Table segmentation

<img src="table_example.jpg"  width="40%" height="40%">

The code contains several functions used for detecting table lines as well as content elements from tables in digitized document images. The documentation of the code is provided in a separate `documentation.pdf` file, and more detailed information on many of the functions and parameters can be found from the documentation of the [OpenCV](https://opencv.org/) library. The code has been developed specifically to detect table lines and content elements from Finnish ship logbooks from the late 19th - early 20th centuries (see example image above), and the default parameter values are optimized for that dataset. 

## Installation

- Create and activate conda environment:

`conda create -n table_env python=3.11`

`conda activate table_env`

- Install required libraries:

`pip install -r requirements.txt`

## Running the code
