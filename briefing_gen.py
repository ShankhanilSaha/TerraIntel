import json
import requests

# Constants
API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def load_inputs():
    with open("outputs/tactical_data.json") as f:
        tactical_data = json.load(f)

    with open("analysed_data/terrain_data.jsonl") as f:
        terrain_lines = f.readlines()
        terrain_data = [json.loads(line) for line in terrain_lines]

    with open("outputs/map.html", encoding="utf-8") as f:
        map_html = f.read()

    return tactical_data, terrain_data, map_html


def get_strategy_mode():
    options = ["stealth", "fast", "loud"]
    print("Select a strategy mode:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option.capitalize()}")

    choice = input("Enter the number of your choice: ").strip()
    while choice not in ["1", "2", "3"]:
        choice = input("Invalid choice. Enter 1, 2, or 3: ").strip()

    return options[int(choice) - 1]


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
    strategy_mode = get_strategy_mode()
    prompt = build_prompt(tactical_data, terrain_data, map_html, strategy_mode)
    strategy = get_war_strategy(prompt)

    print("\n==== STRATEGIC WAR PLANS ====")
    print(strategy)


if __name__ == "__main__":
    main()