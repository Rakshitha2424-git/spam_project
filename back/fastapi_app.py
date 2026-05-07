from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import psycopg2

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()

# -----------------------------
# LOAD ML MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# NEON DATABASE CONNECTION
# -----------------------------
DATABASE_URL = "postgresql://neondb_owner:npg_HZats7yIuWg8@ep-billowing-tree-aq0hg8c5-pooler.c-8.us-east-1.aws.neon.tech/spam-db?sslmode=require&channel_binding=require"

def get_conn():
    return psycopg2.connect(DATABASE_URL)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Message(BaseModel):
    message: str

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}

# -----------------------------
# PREDICT + SAVE
# -----------------------------
@app.post("/predict")
def predict(data: Message):

    message = data.message

    # ML prediction
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    # Save to Neon DB
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO messages (message, prediction) VALUES (%s, %s)",
            (message, result)
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        return {"error": str(e)}

    return {
        "message": message,
        "prediction": result
    }

# -----------------------------
# HISTORY
# -----------------------------
@app.get("/history")
def history():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM messages ORDER BY id DESC")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {"data": rows}