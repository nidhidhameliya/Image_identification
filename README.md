# Image_identification

This project aims to identify whether an image is of a laptop or a mobile device using a Neural Network (NN). The model will be trained on labeled image data to accurately classify new images.

## Overview
This project involves building a convolutional neural network (CNN) to classify images into two categories: laptops and mobiles. The model is trained on a dataset of images and is used to predict whether a given image is of a laptop or a mobile. The project is implemented using TensorFlow and Keras, and it utilizes image augmentation and preprocessing techniques to enhance the model's performance.

## Project Structure
- `proj-1.py`: Main script for running the training and prediction.
- `training/`: Directory containing training images.
- `validation/`: Directory containing validation images.
- `testing/`: Directory containing test images for predictions.
- `model/`: Saved model after training

## Features
- **Data Augmentation**: Augments training data by applying transformations like shear, zoom, and horizontal flipping.
- **CNN Model**: A convolutional neural network is used to classify images into two classes: laptop and mobile.
- **Model Evaluation**: The model is evaluated using accuracy, loss, and visualizations during training.
- **Image Prediction**: Once the model is trained, it can classify new images as laptops or mobiles based on the model's prediction.

## Setup
### Requirements
To set up this project, the following Python libraries need to be installed:
- **tensorflow**
- **numpy**
- **matplotlib**
- **PIL** (for image processing)
- **os** (for path handling)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nidhidhameliya/Image_identification.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training
1. To train the model, run the following command:
   ```bash
   python proj-1.py
   ```
2. The training process will start, and the model will be trained using the images in the `training/` and `validation/` directories. The model's progress will be displayed, and after training, it will be saved as `laptop_mobile_classifier.keras`.

### Running Predictions
1. Once the model is trained, you can use the following function to predict the class of new images:
   ```python
   predict_image(image_path, model)
   ```
   Replace `image_path` with the path to the image you want to classify.

2. Example usage:
   ```python
   image_path = 'C:/Users/Nidhi Dhameliya/Desktop/jupyter/imagedata/testing/1.png'
   predict_image(image_path, model)
   ```

## Results
The model is trained for 20 epochs with an initial training accuracy of around 16.67%, which improves over time. The training results include:
- **Accuracy**: The model's accuracy improves across epochs.
- **Loss**: The model's loss reduces as it learns.
- **Predictions**: The model can classify new images as either "mobile" or "laptop" based on the training data.

