import streamlit as st
import requests

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📩 Spam Message Detector")
st.write("Enter a message to check if it's spam.")

message = st.text_area("Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        try:
            url = "http://127.0.0.1:5000/predict"
            response = requests.post(url, json={"message": message})

            if response.status_code == 200:
                result = response.json()["prediction"]

                if result == "Spam":
                    st.error("🚨 This is Spam")
                else:
                    st.success("✅ Not Spam")
            else:
                st.error("API error")

        except:
            st.error("⚠️ Backend not running")