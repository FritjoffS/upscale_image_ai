# Image Super-Resolution with RealESRGAN

## Description

This Python script implements image super-resolution using the RealESRGAN (Real-World Enhanced Super-Resolution Generative Adversarial Network) model. It takes a low-resolution image as input and generates a high-resolution version of the image.

## Features

- Downloads RealESRGAN model weights automatically
- Supports both CPU and CUDA-enabled GPU processing
- Upscales images by a factor of 4x
- Handles RGB images

## Requirements

- Python 3.8
- Conda

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/realesrgan-superresolution.git
   cd realesrgan-superresolution
Create and activate a Conda environment using the provided environment.yml file:

    bash
    conda env create -f environment.yml
    conda activate realesrgan-env
    

Usage
To upscale an image, run the following command:

    bash
    python upscale.py --input low_res_image.jpg --output high_res_image.jpg
    
Arguments
--input: Path to the input low-resolution image.
--output: Path to save the output high-resolution image.
--cpu: (Optional) Use CPU for processing. If not specified, the script will use a CUDA-enabled GPU if available.
Example

    bash
    python upscale.py --input example.jpg --output example_high_res.jpg 
    
This command will read example.jpg, upscale it by a factor of 4 using the RealESRGAN model, and save the high-resolution image as example_high_res.jpg.

Configuration
The script uses a config.yaml file for configuring various parameters like model weights URL, paths, and RealESRGAN parameters. Ensure this file is properly configured before running the script.

Acknowledgments
This project uses the RealESRGAN model. Special thanks to the authors for their excellent work.

License
This project is licensed under the MIT License. See the LICENSE file for details.

´´´arduino

Make sure to replace `https://github.com/yourusername/realesrgan-superresolution.git` with the actual URL of your repository. This `README.md` provides comprehensive information about your project, including the environment setup using the `environment.yml` file.