from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import requests
import os

# === Constants ===
API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

app = FastAPI()

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Models ===
class StrategyRequest(BaseModel):
    strategy_mode: str

class ChatRequest(BaseModel):
    chat_history: list
    user_message: str

# === Helper Functions ===
def load_inputs():
    with open("outputs/tactical_data.json") as f:
        tactical_data = json.load(f)

    with open("analysed_data/terrain_data.jsonl") as f:
        terrain_lines = f.readlines()
        terrain_data = [json.loads(line) for line in terrain_lines]

    with open("outputs/map.html", encoding="utf-8") as f:
        map_html = f.read()

    return tactical_data, terrain_data, map_html

def build_prompt(tactical_data, terrain_data, map_html, strategy_mode):
    prompt = f"""
You are an advanced military strategy assistant AI.

The following inputs have been provided:
1. Tactical Paths:
    - Easy Path: {len(tactical_data.get("paths", {}).get("easy", []))} coordinates
    - Balanced Path: {len(tactical_data.get("paths", {}).get("balanced", []))} coordinates
    - Tough Path: {len(tactical_data.get("paths", {}).get("tough", []))} coordinates

2. Terrain Data:
    - Total segments: {len(terrain_data)}
    - Sample segment: {terrain_data[0]}

3. Map Markers:
    The map includes hiding spots, surveillance points, and choke points rendered using Leaflet in `map.html`.

Mission Mode: {strategy_mode.upper()}

Based on these inputs, generate 5 distinct, extremely elaborated strategic WAR PLANS optimized for a {strategy_mode.upper()} approach. For each plan, include:
- Objective
- Recommended path (easy/balanced/tough) and justification
- Use of hiding spots and surveillance points
- Expected ambush or conflict zones
- Counter-strategy recommendation
- Terrain advantages or risks

Label the plans as PLAN 1 through PLAN 5. Each should be clear, detailed, and actionable.
"""
    return prompt

def get_chat_response(chat_history):
    body = {
        "model": MODEL,
        "messages": chat_history,
        "temperature": 0.7
    }

    response = requests.post(GROQ_URL, headers=HEADERS, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === API Endpoints ===
@app.post("/generate-plans")
def generate_plans(request: StrategyRequest):
    try:
        tactical_data, terrain_data, map_html = load_inputs()
        prompt = build_prompt(tactical_data, terrain_data, map_html, request.strategy_mode)

        chat_history = [
            {"role": "system", "content": "You are a strategic military planner AI."},
            {"role": "user", "content": prompt}
        ]

        response = get_chat_response(chat_history)
        chat_history.append({"role": "assistant", "content": response})

        return {
            "plans": response,
            "chat_history": chat_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def continue_chat(request: ChatRequest):
    try:
        print("Received user message:", request.user_message)
        print("Chat history length:", len(request.chat_history))

        updated_history = request.chat_history + [
            {"role": "user", "content": request.user_message}
        ]

        reply = get_chat_response(updated_history)
        updated_history.append({"role": "assistant", "content": reply})

        print("Reply from model:", reply[:200])  # Preview only

        return {
            "reply": reply,
            "chat_history": updated_history
        }
    except Exception as e:
        print("Error in /chat:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

