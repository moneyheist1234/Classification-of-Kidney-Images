# Medical Image Classification using Convolutional Neural Networks (CNN)
# Overview
This Python script demonstrates the implementation of a Convolutional Neural Network (CNN) to classify medical images into four categories: Normal, Tumor, Cyst, and Stone. The script uses TensorFlow and Keras libraries for model building, training, and prediction.

# For datasets please search in kaggle like  follow this link to download the realted dataset 

"https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"

# Dependencies
Python 3.x
TensorFlow
Keras
OpenCV (cv2)
NumPy
Matplotlib
PIL (Python Imaging Library)

# Install required libraries:

pip install tensorflow opencv-python numpy matplotlib pillow
Usage
# Prepare your dataset:

Organize your dataset into folders for 'Normal', 'Tumor', 'Cyst', and 'Stone' images.
# Run the code:

Adjust the image_directory variable in the script to point to your dataset directory.
Execute the script to build and train the CNN model.
Test the trained model:

Replace the test_image_path variable in the script with the path to your test image.
Run the code to predict the class of the test image.
Code Structure
Importing Libraries: Import necessary libraries for image processing, model building, and visualization.
Define Image Directories: Specify directories for different image categories - Normal, Tumor, Cyst, and Stone.
Loading and Preprocessing Images: Load images, resize them to the desired input size, and prepare the dataset for model training.
Model Building: Construct a CNN model using Keras Sequential API with Convolutional, Pooling, Flatten, and Dense layers.
Model Training: Compile and train the model using the prepared dataset.
Testing the Model: Load a test image, preprocess it, make predictions, and display the predicted class using Matplotlib.
