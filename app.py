import streamlit as st
import pickle

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

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