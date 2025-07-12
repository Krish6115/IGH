import os
import pandas as pd
import requests
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from googletrans import Translator

# ── CONFIGURATION ─────────────────────────────────────────────────────
GEMINI_API_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
GEMINI_API_KEY  = "AIzaSyC8p47vwjD_0UOVKWEgOBzh5B-vQ3UZ6-w"
EMBED_PATH      = "krishimitra_unified_embeddings.pt"
DATAFILES = {
    "qa":        "krishimitra_dataset.csv",
    "crop_rec":  "city_crop_recommendation.csv",
    "soil":      "soil_health_enhanced.csv"
}

# ── LOAD & PREPARE DATA ────────────────────────────────────────────────
def load_and_unify_qa():
    qa_df   = pd.read_csv(DATAFILES["qa"]).dropna()
    crop_df = pd.read_csv(DATAFILES["crop_rec"]).dropna()
    rows = []
    for _, r in crop_df.iterrows():
        q = (f"Which crop is best for {r.City}, {r.State} "
             f"(soil={r.Soil_Type}, climate={r.Climate})?")
        a = r.Recommended_Crop
        rows.append({"questions": q, "answers": a})
    return pd.concat([qa_df[["questions","answers"]], pd.DataFrame(rows)], ignore_index=True)

def load_soil():
    return pd.read_csv(DATAFILES["soil"]).dropna()

# ── INITIALIZE MODELS & EMBEDDINGS ────────────────────────────────────
def init():
    unified = load_and_unify_qa()
    soil    = load_soil()
    st      = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    texts   = unified.questions.tolist()

    if os.path.exists(EMBED_PATH):
        emb = torch.load(EMBED_PATH)
    else:
        emb = st.encode(texts, convert_to_tensor=True)
        torch.save(emb, EMBED_PATH)

    tr = Translator()
    return unified, soil, st, emb, tr

_unified_df, _soil_df, _st_model, _embeddings, _translator = init()

# ── FLASK APP ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="No text provided"), 400

    # 1) Detect language & translate to English
    lang = detect(text)
    text_en = text if lang=='en' else _translator.translate(text, dest='en').text

    # 2) Semantic QA retrieval
    q_emb   = _st_model.encode(text_en, convert_to_tensor=True)
    scores  = util.pytorch_cos_sim(q_emb, _embeddings)[0]
    idx     = torch.argmax(scores).item()
    qa_ans  = _unified_df.answers.iloc[idx]

    # 3) Match soil by district if mentioned, else random
    soil_fact = None
    for d in _soil_df.District:
        if d.lower() in text_en.lower():
            soil_fact = _soil_df[_soil_df.District.str.lower()==d.lower()].iloc[0].to_dict()
            break
    if soil_fact is None:
        soil_fact = _soil_df.sample(1).iloc[0].to_dict()

    # 4) Build Gemini prompt
    prompt = (
        f"You are an intelligent agricultural assistant.\n"
        f"Use this QA snippet: {qa_ans}\n"
        f"Soil data: {soil_fact}\n"
        f"User asks: {text_en}\n"
        f"Answer concisely."
    )

    # 5) Call Gemini with error handling
    try:
        resp = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            json={"contents":[{"parts":[{"text": prompt}]}]},
            headers={"Content-Type":"application/json"}
        )
        resp.raise_for_status()
        answer_en = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as http_err:
        if resp.status_code == 429:
            answer_en = "Sorry, the service is temporarily busy. Please try again in a moment."
        else:
            answer_en = "Sorry, an unexpected error occurred. Please try again later."
    except Exception:
        answer_en = "Sorry, I couldn't reach the AI service. Please check your connection."

    # 6) Translate back if needed
    final = answer_en if lang=='en' else _translator.translate(answer_en, dest=lang).text
    return jsonify(reply=final, lang=lang)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
