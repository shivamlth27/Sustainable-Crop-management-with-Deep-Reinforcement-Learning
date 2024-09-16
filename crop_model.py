import pandas as pd
import torch
import torch.nn as nn
from cr import QLearningModel  # Import your actual model class from the model file

# Load the saved scalar and label encoder with map_location for CPU compatibility
scalar = torch.load('scaler.pkl', map_location=torch.device('cpu'))
label_encoder = torch.load('label_encoder.pkl', map_location=torch.device('cpu'))

# Load the saved model state dict with map_location for CPU compatibility
model_state_dict = torch.load('model.pth', map_location=torch.device('cpu'))

# Create an instance of your model class
# Ensure input_size and num_classes are correct for your model
model = QLearningModel(7, 22)  

# Load the model state dict
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Define the preprocess_input function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    scaled_data = scalar.transform(df)
    return torch.tensor(scaled_data, dtype=torch.float32)

# Define the predict function
def predict(input_data):
    with torch.no_grad():
        input_tensor = preprocess_input(input_data)
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1)
        # Ensure label_encoder can handle the prediction
        return label_encoder.inverse_transform(predicted_class.numpy())

# Test the predict function with sample input data
sample_input = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 21,
    'humidity': 82,
    'ph': 6.5,
    'rainfall': 203
}

prediction = predict(sample_input)
print('Predicted crop:', prediction[0])  # Assuming prediction is a single value
