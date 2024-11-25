import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = torch.load('/Users/anody/Downloads/fish_model.pth', map_location=torch.device('cpu'))  # Adjust path if needed
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
    transforms.Resize((224, 224)),  # Match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

# Define class names
class_names = [
    "Fresh Day 1 Eyes", "Fresh Day 1 Gills",
    "Non-Fresh Day 2 Eyes", "Non-Fresh Day 2 Gills",
    "Non-Fresh Day 3 Eyes", "Non-Fresh Day 3 Gills",
    "Non-Fresh Day 4 Eyes", "Non-Fresh Day 4 Gills"
]

# Streamlit app interface
st.title("Fish Freshness Detection from Uploaded Image")
st.markdown("""
1. Upload an image of the fish.
2. The model will predict the freshness stage and part of the fish.
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image of the fish", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
        prediction = class_names[predicted_class.item()]  # Get the class name

    # Display prediction
    st.write(f"**Prediction:** {prediction}")
