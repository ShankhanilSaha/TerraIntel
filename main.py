import json
import requests

# === Constants ===
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

Label the plans as PLAN 1 through PLAN 5. Each plan must be clear, detailed, and actionable.

**IMPORTANT:** Format all output in a neat, easy-to-read table with proper headers and alignment, suitable for direct use in reports or presentations.
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

def main():
    tactical_data, terrain_data, map_html = load_inputs()

    print("Choose your strategy mode (stealthy / fast / loud):")
    strategy_mode = input("> ").strip().lower()
    if strategy_mode not in ["stealthy", "fast", "loud"]:
        print("Invalid strategy mode. Defaulting to 'stealthy'.")
        strategy_mode = "stealthy"

    prompt = build_prompt(tactical_data, terrain_data, map_html, strategy_mode)

    chat_history = [
        {"role": "system", "content": "You are a strategic military planner AI."},
        {"role": "user", "content": prompt}
    ]

    print("\nGenerating war plans... please wait...\n")
    plans_response = get_chat_response(chat_history)
    chat_history.append({"role": "assistant", "content": plans_response})

    print(plans_response)
    print("\n--- War plans generated. You can now chat with the AI. Type 'exit' to quit. ---\n")

    # Chat loop
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        chat_history.append({"role": "user", "content": user_message})

        try:
            reply = get_chat_response(chat_history)
            print(f"AI: {reply}\n")
            chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"Error during chat response: {e}")
            break

if __name__ == "__main__":
    main()
