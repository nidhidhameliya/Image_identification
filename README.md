# Image_identification

This project aims to identify whether an image is of a laptop or a mobile device using a Neural Network (NN). The model will be trained on labeled image data to accurately classify new images.

## Project Description

- Objective: To build a Neural Network model that can distinguish between images of laptops and mobiles.
- Data: The dataset consists of labeled images that are categorized as either laptops or mobiles.
- Methodology:
  - Load and preprocess the labeled image data.
  - Train a Convolutional Neural Network (CNN) on the preprocessed images.
  - Use the trained model to predict whether a new image is a laptop or a mobile.

## Instructions

1. Data Preparation:
   - Ensure all labeled images are placed correctly in the dataset directory.
   - The dataset should be structured such that each category (laptop or mobile) has its own subfolder containing respective images.

2. Model Training:
   - Use the Jupyter notebook provided (`project-1.ipynb`) to run the model training steps.
   - The notebook guides you through loading the data, preprocessing it, and training the Neural Network.

3. Model Prediction:
   - After training, you can use the model to classify new images by following the instructions in the notebook.

## Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Libraries: TensorFlow, OpenCV, NumPy, Pillow

## Installation

Install the necessary libraries using pip:
pip install tensorflow opencv-python numpy pillow

## Usage

1. Clone the repository:
   git clone https://github.com/nidhidhameliya/image_identification.git
   cd image_identification

2. Start Jupyter Notebook:
   jupyter notebook

3. Open `project-1.ipynb` and follow the instructions to train the model and make predictions.

