import os
import json
import requests

# Constants
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def load_inputs():
    with open("analysed_data/tactical_data_path.json") as f:
        tactical_data = json.load(f)

    with open("analysed_data/terrain_data.jsonl") as f:
        terrain_lines = f.readlines()
        terrain_data = [json.loads(line) for line in terrain_lines]

    with open("outputs/map.html",encoding = "utf-8") as f:
        map_html = f.read()

    return tactical_data, terrain_data, map_html

def build_prompt(tactical_data, terrain_data, map_html):
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

Based on these inputs, generate a strategic WAR PLAN which includes:
- Objective
- Recommended path (easy/balanced/tough) and justification
- Use of hiding spots and surveillance points
- Expected ambush or conflict zones
- Counter-strategy recommendation
- Terrain advantages or risks

Give a clear, concise, and actionable plan.
"""
    return prompt

def get_war_strategy(prompt):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a strategic military planner AI."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(GROQ_URL, headers=HEADERS, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    if not API_KEY:
        raise EnvironmentError("GROQ_API_KEY not found in environment.")

    tactical_data, terrain_data, map_html = load_inputs()
    prompt = build_prompt(tactical_data, terrain_data, map_html)
    strategy = get_war_strategy(prompt)

    print("\n==== STRATEGIC WAR PLAN ====\n")
    print(strategy)

if __name__ == "__main__":
    main()
