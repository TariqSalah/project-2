import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load the pre-trained ResNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Define ImageNet classes
with open('/workspaces/Fashion-MNIST-Web/imagenet-classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to make predictions using ResNet
def predict_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_label = classes[predicted.item()]
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        if image_file:
            # Save the image temporarily
            image_path = 'temp_image.png'
            image_file.save(image_path)
            # Get the prediction
            prediction = predict_image(image_path)
            return render_template('result.html', prediction=prediction)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
