# Import necessary libraries
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
from PIL import Image
import io
import IPython.display as display
from ipywidgets import FileUpload, Button, Output, VBox, Label

# Load the pre-trained face mask detection model
mymodel = load_model('mymodel.h5')

# Define the face detection model (Haarcascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize image to match the input shape of the model
    img = image.img_to_array(img)  # Convert to array
    img = np.expand_dims(img, axis=0)  # Add an extra dimension
    img = img / 255.0  # Normalize the image (same preprocessing as training)
    return img

# Function to make predictions using the loaded model
def predict_mask(img):
    faces = face_cascade.detectMultiScale(np.array(img), scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        face_img = np.array(img)[y:y+h, x:x+w]
        face_img = Image.fromarray(face_img).resize((150, 150))
        face_array = preprocess_image(face_img)
        prediction = mymodel.predict(face_array)[0][0]  # 1 = No Mask, 0 = Mask
        
        # Annotate the image with the prediction result
        if prediction == 1:
            cv2.rectangle(np.array(img), (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(np.array(img), 'NO MASK', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(np.array(img), (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(np.array(img), 'MASK', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert image back to displayable format
    img_disp = Image.fromarray(np.array(img))
    display.display(img_disp)

# Function to handle file uploads and predictions
def on_file_upload(change):
    uploaded_file = list(upload_widget.value.values())[0]
    img = Image.open(io.BytesIO(uploaded_file['content']))
    
    # Display the uploaded image
    display.clear_output(wait=True)
    display.display(img)
    
    # Make the prediction on the uploaded image
    predict_mask(img)

# Widget to upload image
upload_widget = FileUpload(accept='image/*', multiple=False)
upload_widget.observe(on_file_upload, names='value')

# Display the upload widget
display.display(Label('Upload an image for face mask detection:'))
display.display(upload_widget)
