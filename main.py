from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import json


app = FastAPI()


class Crop(BaseModel):
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    nitrogen: float
    phosphorous: float
    potassium: float
    

@app.get("/")
def crop():
    return {"message": "Hello World"}


@app.post("/predict")
def predict(data: Crop):
    
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f) 
                
    data_dict = data.dict()
    print(data_dict)
    
    features = [data_dict['temperature'], data_dict['humidity'], data_dict['ph'], data_dict['rainfall'], data_dict['nitrogen'], data_dict['phosphorous'], data_dict['potassium']]
    crop_names = {
                1: "Rice", 2: "maize", 3: "chikpea", 4: "kidneybeans", 5: "pigeonpeas",
                6: "mothbeans", 7: "mungbeans", 8: "blackgrams", 9: "lentil", 10: "pomogranate",
                11: "banana", 12: "mango", 13: "grapes", 14: "watermelon", 15: "muskmeelon",
                16: "apple", 17: "orange", 18: "papaya", 19: "coconut", 20: "cotton", 21: "jute", 22: "coffee"
            }
    
    prediction = model.predict(np.array([features]))
    pred = crop_names[prediction[0]]
    
    return {"prediction": pred}
    
    

if __name__ == "__main__":
    app.run()
    