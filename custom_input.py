import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load the pre-trained model
mymodel = tf.keras.models.load_model('mymodel.h5')

# Function to predict if an individual is wearing a mask
def predict_mask(img_path):
    # Load the image and resize it to the target size (150, 150, 3)
    img = image.load_img(img_path, target_size=(150, 150, 3))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model's expected input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (if needed) to match the training data (scaling by 1/255)
    img_array /= 255.0
    
    # Predict using the model
    prediction = mymodel.predict(img_array)[0][0]
    
    # Interpret the prediction
    if prediction == 1:
        result = "No Mask"
    else:
        result = "Mask"
    
    return result

# Prompt the user to input the image path
img_path = input("Please enter the full path to the image file: ")

# Validate if the provided path exists
if os.path.exists(img_path):
    # Run the prediction
    result = predict_mask(img_path)
    print(f"Prediction: {result}")

    # (Optional) Display the image with OpenCV
    img = cv2.imread(img_path)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: The provided image path does not exist. Please check and try again.")
    
