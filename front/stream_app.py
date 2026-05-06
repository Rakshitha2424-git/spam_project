import streamlit as st
import requests

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📩 Spam Message Detector")
st.markdown("Check whether a message is **Spam or Not Spam** using ML")

# Input box
message = st.text_area("Enter your message:", height=150)

# Button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        with st.spinner("Analyzing message... ⏳"):
            try:
                url = "http://127.0.0.1:8000/predict"
                response = requests.post(url, json={"message": message})

                if response.status_code == 200:
                    result = response.json()["prediction"]

                    st.subheader("Result:")

                    if result == "Spam":
                        st.error("🚨 This message is Spam")
                    else:
                        st.success("✅ This message is Not Spam")

                else:
                    st.error("❌ API Error")

            except:
                st.error("⚠️ Backend not running")