# Crop Prediction API

This is a RESTful API built with FastAPI that predicts the type of crop based on environmental and soil attributes using a machine learning model. The model is trained on historical data to provide accurate predictions.

Predicts the type of crop based on environmental and soil attributes.
Uses a machine learning ensemble model (Voting Classifier) trained on agricultural data.
RESTful API endpoints for easy integration with other applications.

## Dependencies
```
Python 3.7+
FastAPI 0.68.1
scikit-learn (for model training and prediction)
uvicorn (for running the FastAPI application server)
pydantic (for data validation with FastAPI)
numpy (for numerical operations)
```
## Installation
```
Clone the repository:

bash

Copy code
git clone https://github.com/vedant0321/crop-api
cd crop-prediction-api
Install dependencies:

bash

Copy code
pip install -r requirements.txt
Usage
Running the API
Start the FastAPI application server using uvicorn:

bash

Copy code
uvicorn main:app --reload
The API will start running at http://127.0.0.1:8000.
```
# Making Predictions

Once the API is running, you can make POST requests to the /predict endpoint with JSON data containing environmental and soil attributes. The API will return the predicted crop type.

# Endpoints
- GET /: Returns a simple welcome message.
- POST /predict: Endpoint to make crop predictions based on input data.
- Input Data
    - The /predict endpoint expects JSON data with the following fields:

        - temperature: Temperature in Celsius.
        - ph: Soil pH level.
        - humidity: Relative humidity in percentage.
        - rainfall: Rainfall in mm.
        - nitrogen: Nitrogen content in soil.
        - phosphorous: Phosphorous content in soil.
        - potassium: Potassium content in soil.
- Output
The /predict endpoint returns JSON with the predicted crop type:
```
{
      "prediction": "Rice"
}
```

```
Examples

Example 1: Python Requests:

import requests

url = 'http://127.0.0.1:8000/predict'
data = {
    "temperature": 25.6,
    "humidity": 78.2,
    "ph": 6.5,
    "rainfall": 62.3,
    "nitrogen": 45.6,
    "phosphorous": 36.7,
    "potassium": 22.4
}

response = requests.post(url, json=data)
print(response.json())  # {"prediction": "Rice"}
```





