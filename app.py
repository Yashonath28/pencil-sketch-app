from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

def convert_to_sketch(image_path, shadow_intensity):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)
    
    # Apply Gaussian Blur to the inverted image
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    
    # Blend the grayscale image and the blurred inverted image
    sketch = cv2.divide(gray_image, 255 - blurred, scale=256.0 - shadow_intensity)
    
    # Use adaptive thresholding to create a pencil sketch effect with enhanced edges
    sketch = cv2.adaptiveThreshold(sketch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 9, 10)
    
    return sketch

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['imageFile']
    shadow_intensity = float(request.form['shadow'])  # Get shadow intensity from slider
    image = np.array(Image.open(file))
    temp_filename = "temp_image.jpg"
    cv2.imwrite(temp_filename, image)
    sketch = convert_to_sketch(temp_filename, shadow_intensity)
    _, img_encoded = cv2.imencode('.jpg', sketch)
    img_bytes = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': img_bytes})

if __name__ == '__main__':
    app.run(debug=True)
