import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pickle
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

# Load models for plant disease prediction
plant_disease_model = tf.keras.models.load_model('plant_disease_model.h5')
class_indices = json.load(open('class_indices.json'))
class_indices = {int(k): v for k, v in class_indices.items()}  # Ensure keys are int

# Load models for crop recommendation
with open("model_useing_Location_Seasion_Area.pkl", "rb") as f:
    location_model = pickle.load(f)

with open("model_useing_npk.pkl", "rb") as f:
    npk_model = pickle.load(f)

# Load unique values from the JSON file
with open("unique_values.json", "r") as f:
    unique_values = json.load(f)

class CropRequest(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    season: Optional[str] = None
    area: Optional[float] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None

def fetch_gemini_advice(crop, disease):
    """Fetches step-by-step cure instructions for the given crop and disease using Google Gemini API."""
    if disease.lower() == "healthy":
        return "The plant is healthy. No action needed."
    
    base_prompt = (
        f"Provide only step-by-step instructions to cure {disease} in {crop}. "
        "Do not include any introduction or description, just list the steps clearly and concisely."
    )
    response = gemini_model.generate_content(base_prompt)
    return response.text if response else "No cure information found."

def parse_prediction(prediction):
    """Separates the crop name and disease name from the prediction."""
    if "___" in prediction:
        crop, disease = prediction.split("___")
    else:
        crop, disease = prediction, "Unknown"
    return crop, disease

# Image Preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array

# Function to Convert Image to JPG
def convert_to_jpg(image_path: str) -> str:
    """
    Converts any image to JPG format and returns the path to the converted image.
    """
    # Open the image
    img = Image.open(image_path)

    # Ensure the image is in RGB mode (important for non-RGB images like PNG)
    img = img.convert("RGB")

    # Define the new file path (adding .jpg extension)
    new_image_path = os.path.splitext(image_path)[0] + ".jpg"

    # Save the image as JPG
    img.save(new_image_path, "JPEG")

    # Return the path of the new image
    return new_image_path

# Prediction Function for Plant Disease
def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = plant_disease_model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[predicted_class_index]

@app.post('/predict_disease')
async def predict_disease(file: UploadFile = File(...)):
    # Save the uploaded file to /tmp directory
    file_path = os.path.join('/tmp', file.filename)
    with open(file_path, 'wb') as buffer:
        buffer.write(await file.read())

    # Convert the image to JPG if it's not already in JPG format
    if not file.filename.lower().endswith(".jpg"):
        file_path = convert_to_jpg(file_path)

    # Predict the class
    prediction = predict_image_class(file_path)
    crop, disease = parse_prediction(prediction)

    # Get cure steps from Google Gemini
    cure_steps = fetch_gemini_advice(crop, disease)

    # Remove the temporary file
    os.remove(file_path)

    return {
        "crop": crop,
        "predicted_disease": disease,
        "cure_steps": cure_steps
    }

# Helper function to get top recommendations from the model
def get_top_recommendations(model, input_data, top_n=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)
        top_indices = np.argsort(proba[0])[-top_n:][::-1]
        return [model.classes_[i] for i in top_indices]
    return model.predict(input_data).tolist()

@app.post("/predict_crop")
def predict_crop(request: CropRequest):
    location_recommendations = []
    npk_recommendations = []

    if request.state and request.district and request.season and request.area:
        try:
            input_data = pd.DataFrame([[request.state, request.district, request.season, np.log1p(request.area)]],
                                      columns=['State_Name', 'District_Name', 'Season', 'Area'])
            location_recommendations = get_top_recommendations(location_model, input_data, top_n=3)
        except ValueError:
            pass  # Handle invalid input

    if request.N is not None and request.P is not None and request.K is not None and request.temperature is not None \
            and request.humidity is not None and request.ph is not None and request.rainfall is not None:
        try:
            input_data = np.array([[request.N, request.P, request.K, request.temperature,
                                    request.humidity, request.ph, request.rainfall]])
            npk_recommendations = get_top_recommendations(npk_model, input_data, top_n=3)
        except ValueError:
            pass  # Handle invalid input

    location_lower = {crop.lower(): crop for crop in location_recommendations}
    common_crops_lower = set(location_lower.keys()) & set(crop.lower() for crop in npk_recommendations)
    common_crops = [location_lower[crop] for crop in common_crops_lower]

    location_recommendations = [crop for crop in location_recommendations if crop.lower() not in common_crops_lower]
    npk_recommendations = [crop for crop in npk_recommendations if crop.lower() not in common_crops_lower]

    merged_crops = common_crops + npk_recommendations + location_recommendations
    final_recommendations = [crop.lower() for crop in merged_crops[:4]]

    return {"recommendations": final_recommendations if final_recommendations else []}




@app.get("/")
def read_root():
    return {"message": "API is running!"}
