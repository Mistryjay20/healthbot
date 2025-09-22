from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os

# ----------------------------
# Load CSV
# ----------------------------
CSV_PATH = "Dataset/disease2.csv"
df = pd.read_csv(CSV_PATH)

disease_symptoms = {}
disease_precautions = {}

for _, row in df.iterrows():
    disease = row["Disease"].strip().lower()
    symptoms = [str(row[col]).strip() for col in df.columns if col.startswith("Symptom") and pd.notna(row[col])]
    precautions = [str(row[col]).strip() for col in df.columns if col.startswith("Precaution") and pd.notna(row[col])]
    disease_symptoms[disease] = symptoms
    disease_precautions[disease] = precautions

# ----------------------------
# Gemini
# ----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))   # 🔑 set your API key as env variable
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# Allow frontend (index.html) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/message")
async def message(req: Message):
    user_text = req.text.lower().strip()

    # Greeting
    if user_text in ["hi", "hello", "hey"]:
        return {"reply": "👋 Welcome to Health Awareness Assistant!\n\n1. 📝 Check symptoms\n2. 🛡 Precautions\n3. ℹ General health tips\n\nHow can I help you today?"}

    # 1️⃣ Match symptoms
    for disease, symptoms in disease_symptoms.items():
        if any(symptom.replace("_", " ") in user_text for symptom in symptoms):
            reply = f" Possible condition: **{disease.title()}**\n\n"
            reply += " Symptoms:\n" + "\n".join([f"{i+1}. {s.replace('_',' ')}" for i, s in enumerate(symptoms)])
            precautions = disease_precautions.get(disease, [])
            if precautions:
                reply += "\n\n Precautions:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(precautions)])
            reply += "\n\n This is awareness info only. Please consult a doctor."
            return {"reply": reply}

    # 2️⃣ Match disease name directly
    for disease, precautions in disease_precautions.items():
        if disease in user_text:
            reply = f" Precautions for **{disease.title()}**:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(precautions)])
            reply += "\n\n This is awareness info only. Please consult a doctor."
            return {"reply": reply}

    # 3️⃣ Fallback → Gemini AI
    try:
        response = gemini_model.generate_content(
            f"You are a health awareness assistant. User asked: {req.text}. "
            f"Reply in short **numbered points**, clear and friendly."
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"⚠️ AI error: {str(e)}"}


@app.get("/")
async def root():
    return {"message": "Health Awareness API running ✅"}
