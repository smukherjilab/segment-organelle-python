# Segment and Measure Organelles in Cells

This repo contains scripts that I use to measure the sizes and numbers of different organelles in cells, from microscopy images to 

## Workflow:

- step 1: segment the cell
    - main script: `call_yeaz.py`
    - input: bright field microscopy image
    - output: integer label images
    - other files: 
        - `better_find_cells.py` was a project that did not make sense. 
        - `find_affine_transform.py` was used to find the affine transform from the Bright field images to the spectral images. Its result has been incorporated into the main script. So you might want to delete them if you are processing fluorescent microscopy images.
- step 2: segment the organelles
- step 3: measure the organelle properties into csv files
- step 4: read the csv files and find insights from the data
    - `analysis_6colors_largeDataset.py`

## Script Structure

For each step, the work can be abtracted as "take some input, apply some manipulations, and save the result into the output", so I wrap the manipulations into a function called `wrapper()`, which takes a tuple `(input,output)` as input, and the function `latsingdan`

- `latsingdan`: 