import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the model once
model = load_model("best_model.h5")

# Define classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

def predict(image_bytes):
    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # normalize
    image = np.expand_dims(image, axis=(0, -1))  # shape: (1, 64, 64, 1)

    # Predict
    prediction = model.predict(image)
    index = np.argmax(prediction)
    label = CLASSES[index]
    confidence = float(prediction[0][index])
    return {"prediction": label, "confidence": round(confidence, 3)}
