from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI(title="Spam Detection API")

# Load model
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# ✅ Define request body (this is new & powerful)
class Message(BaseModel):
    message: str

# Home route
@app.get("/")
def home():
    return {"status": "API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: Message):
    try:
        message = data.message

        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        result = "Spam" if prediction == 1 else "Not Spam"

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))