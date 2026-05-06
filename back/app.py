import streamlit as st
import pickle

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# Home route (GET)
@app.route("/")
def home():
    return "Spam Detection API Running 🚀"

# Predict route (POST)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Check input
    if not data or "message" not in data:
        return jsonify({"error": "Please provide a message"}), 400

    message = data["message"]

    # Transform + predict
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"prediction": result})

# Run server
if __name__ == "__main__":
    app.run(debug=True)
st.title("Spam Detector")

msg = st.text_input("Enter message")

if st.button("Predict"):
    data = vectorizer.transform([msg])
    result = model.predict(data)

    if result[0] == 1:
        st.error("Spam 🚨")
    else:
        st.success("Not Spam ✅")
prob = model.predict_proba(data)[0][1]
st.write(f"Spam Probability: {prob:.2f}")
if msg.strip() == "":
    st.warning("Please enter a message")