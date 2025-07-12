🌾 AgriGuru: Intelligent Agricultural Voice Assistant:
AgriGuru is an AI-powered voice assistant for farmers that delivers personalized agricultural advice using a combination of local datasets and Gemini AI. It supports multilingual voice interaction and responds using speech in the same language.
🧠 Features:

🎙️ Voice-based input with speech recognition

🌐 Multilingual: Understands and replies in user’s spoken language

🌱 Crop recommendations based on city soil/climate

🧪 Soil diagnostics support (pH, NPK, micronutrients)

📡 Fallback to Gemini API for intelligent, up-to-date responses

🗣️ Text-to-speech replies using gTTS

🛠️ Tech Stack:

Layer	Tool / Framework

🐍 Language	Python

🎤 Speech	speechrecognition + Google STT

🔊 Voice Output	gTTS + OS audio playback

🧠 Embeddings	sentence-transformers (MiniLM)

🌍 Translation	googletrans

📡 AI Model	Google Gemini (via REST API)

📊 Data Handling	pandas, torch

📁 Project Structure
bash
📦 agriguru/
├── agriguru_server.py           # Main backend Flask server
├── krishimitra_dataset.csv      # Predefined farming Q&A
├── city_crop_recommendation.csv # Crop suggestions by region
├── soil_health_enhanced.csv     # Soil parameters by district
├── krishimitra_unified_embeddings.pt # Precomputed embeddings
└── index.html                   # (Optional) Frontend UI
⚙️ Installation

bash
pip install flask flask-cors pandas requests sentence-transformers torch langdetect googletrans==4.0.0-rc1 gTTS speechrecognition

⚠️ On Windows, ensure your system has ffmpeg or vlc installed for mp3 playback.

🚀 Running the Voice Assistant
bash :
python agriguru_server.py

It will listen to your voice through the microphone, process your query, and respond in your language using voice and text.

🎤 Example Queries:
"Suggest the best crop to grow in Bangalore"
"What is the ideal soil pH for paddy?"
"What fertilizer should I use for black soil?"
"Which crop grows best in Patna?"
🔐 API Key Setup :
You must embed your Gemini API key in the script like:
GEMINI_API_KEY = "your-api-key"
📝 Acknowledgements:
Google Gemini AI
Google Speech-to-Text and gTTS
Kaggle & government data portals for soil and crop data
-- Its still in the development! So we are trying to integrate with Twilio kinda so it makes the work more easier.
