import numpy as np
from keras.preprocessing import image
from keras.models import load_model

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the trained model
model = load_model('/workspaces/Fashion-MNIST-Web/fashion_mnist_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


# Make predictions
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]
