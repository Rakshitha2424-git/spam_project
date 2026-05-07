import streamlit as st
import requests
import psycopg2
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Spam Detector", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"

DATABASE_URL = "postgresql://neondb_owner:npg_HZats7yIuWg8@ep-billowing-tree-aq0hg8c5-pooler.c-8.us-east-1.aws.neon.tech/spam-db?sslmode=require&channel_binding=require"

# ----------------------------
# TITLE
# ----------------------------
st.title("📩 Spam Message Detector")
st.markdown("Check whether a message is **Spam or Not Spam** using ML")

# ----------------------------
# DB FUNCTION
# ----------------------------
def get_data():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM messages", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()

# ----------------------------
# LOAD DATA
# ----------------------------
df = get_data()

# ----------------------------
# DASHBOARD (only if data exists)
# ----------------------------
if not df.empty:

    st.subheader("📊 Spam vs Not Spam")
    count_df = df["prediction"].value_counts()
    st.bar_chart(count_df)

    st.subheader("📈 Spam Trends Over Time")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    trend = df.groupby(df["created_at"].dt.date)["prediction"].count()
    st.line_chart(trend)

    st.subheader("🔥 Top Spam Messages")

    spam_df = df[df["prediction"] == "Spam"]
    top_messages = spam_df["message"].value_counts().head(10)
    st.dataframe(top_messages)

else:
    st.warning("No data found in database")

# ----------------------------
# INPUT SECTION
# ----------------------------
st.divider()

message = st.text_area("Enter your message:", height=150)

if st.button("Predict"):

    if not message.strip():
        st.warning("⚠️ Please enter a message")

    else:
        with st.spinner("Analyzing message... ⏳"):
            try:
                response = requests.post(API_URL, json={"message": message})

                if response.status_code == 200:
                    result = response.json()["prediction"]

                    st.subheader("Result:")

                    if result == "Spam":
                        st.error("🚨 This message is Spam")
                    else:
                        st.success("✅ This message is Not Spam")

                else:
                    st.error("❌ API Error")

            except Exception as e:
                st.error(f"⚠️ Backend not running: {e}")