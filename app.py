from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import os

# ==========================================
# Create logs folder automatically
# ==========================================
os.makedirs("logs", exist_ok=True)

# ==========================================
# Configure Logging
# ==========================================
logging.basicConfig(
    filename="logs/prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================
# Initialize FastAPI
# ==========================================
app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Predict Airbnb listing prices using Machine Learning",
    version="1.0"
)

# ==========================================
# Load Model & Preprocessor
# ==========================================
model = joblib.load("models/airbnb_price_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# ==========================================
# Input Schema
# ==========================================
class AirbnbInput(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    latitude: float
    longitude: float
    room_type: str
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int
    review_year: float
    review_month: float
    has_review: int


# ==========================================
# Home Route
# ==========================================
@app.get("/")
def home():
    return {
        "message": "Airbnb Price Prediction API is running!"
    }


# ==========================================
# Prediction Route
# ==========================================
@app.post("/predict")
def predict(data: AirbnbInput):

    try:
        # Convert request into DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Preprocess input
        processed_data = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed_data)

        predicted_price = round(float(prediction[0]), 2)

        # Save prediction log
        logging.info(
            f"Prediction Request | "
            f"Neighbourhood: {data.neighbourhood} | "
            f"Room Type: {data.room_type} | "
            f"Predicted Price: {predicted_price}"
        )

        # Return response
        return {
            "status": "success",
            "predicted_price": predicted_price,
            "currency": "USD"
        }

    except Exception as e:

        # Save error log
        logging.error(f"Prediction Error: {str(e)}")

        return {
            "status": "error",
            "message": str(e)
        }