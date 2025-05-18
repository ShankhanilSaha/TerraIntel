from flask import Flask, render_template, request, jsonify
import json
import requests
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# === Constants ===
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Store chat histories and CLI outputs for different sessions
chat_histories = {}
cli_outputs = {}

def generate_tactical_data(lat1, lon1, lat2, lon2):
    """Generate tactical data based on input coordinates"""
    tactical_data = {
        "paths": {
            "easy": [[lat1, lon1], [lat2, lon2]],
            "balanced": [
                [lat1 + 0.001, lon1 + 0.001],
                [lat2 - 0.001, lon2 - 0.001]
            ],
            "tough": [
                [lat1 - 0.001, lon1 - 0.001],
                [lat2 + 0.001, lon2 + 0.001]
            ]
        },
        "tactical_points": {
            "hiding_spots": [
                [lat1 + (lat2 - lat1) * 0.25, lon1 + (lon2 - lon1) * 0.25],
                [lat1 + (lat2 - lat1) * 0.75, lon1 + (lon2 - lon1) * 0.75]
            ],
            "choke_points": [
                [lat1 + (lat2 - lat1) * 0.5, lon1 + (lon2 - lon1) * 0.5]
            ],
            "checkpoints": [
                [lat1 + (lat2 - lat1) * 0.33, lon1 + (lon2 - lon1) * 0.33],
                [lat1 + (lat2 - lat1) * 0.66, lon1 + (lon2 - lon1) * 0.66]
            ]
        }
    }
    
    # Save tactical data to file
    with open("outputs/tactical_data.json", "w") as f:
        json.dump(tactical_data, f, indent=2)
    
    return tactical_data

def load_inputs(lat1=None, lon1=None, lat2=None, lon2=None):
    """Load or generate tactical data and terrain information"""
    if all(x is not None for x in [lat1, lon1, lat2, lon2]):
        tactical_data = generate_tactical_data(lat1, lon1, lat2, lon2)
    else:
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy_endpoint():
    try:
        data = request.json
        strategy_mode = data.get('strategy_mode', 'stealthy')
        lat1 = float(data.get('lat1'))
        lon1 = float(data.get('lon1'))
        lat2 = float(data.get('lat2'))
        lon2 = float(data.get('lon2'))
        
        # Load tactical data and terrain information with coordinates
        tactical_data, terrain_data, map_html = load_inputs(lat1, lon1, lat2, lon2)
        
        # Build the prompt and initialize chat history
        prompt = build_prompt(tactical_data, terrain_data, map_html, strategy_mode)
        chat_history = [
            {"role": "system", "content": "You are a strategic military planner AI."},
            {"role": "user", "content": prompt}
        ]
        
        # Generate the war plans
        plans_response = get_chat_response(chat_history)
        chat_history.append({"role": "assistant", "content": plans_response})
        
        # Store chat history for this session
        session_id = request.headers.get('X-Session-ID')
        if session_id:
            chat_histories[session_id] = chat_history
            
            # Create CLI-style output
            cli_output = [
                "\n=== TerraIntel Strategy Generator ===",
                f"\nStrategy mode selected: {strategy_mode.upper()}",
                f"\nCoordinates:",
                f"  Start: ({lat1}, {lon1})",
                f"  End: ({lat2}, {lon2})",
                "\nGenerating war plans... please wait...\n",
                plans_response,
                "\n--- War plans generated. You can now ask follow-up questions. ---\n"
            ]
            cli_outputs[session_id] = cli_output
        
        return jsonify({
            'success': True,
            'strategy': plans_response,
            'cli_output': '\n'.join(cli_outputs.get(session_id, []))
        })
            
    except Exception as e:
        print(f"Error generating strategy: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data['question']
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in chat_histories:
            return jsonify({
                'success': False,
                'error': 'No active strategy session. Please generate a strategy first.'
            })
        
        # Get the chat history for this session
        chat_history = chat_histories[session_id]
        cli_output = cli_outputs.get(session_id, [])
        
        # Add question to CLI output
        cli_output.append(f"You: {question}")
        
        if question.lower() == 'exit':
            cli_output.append("Chat session ended.")
            cli_outputs[session_id] = cli_output
            return jsonify({
                'success': True,
                'response': "Chat session ended.",
                'cli_output': '\n'.join(cli_output)
            })
        
        # Get AI response
        chat_history.append({"role": "user", "content": question})
        response = get_chat_response(chat_history)
        chat_history.append({"role": "assistant", "content": response})
        
        # Update the stored chat history and CLI output
        chat_histories[session_id] = chat_history
        cli_output.append(f"AI: {response}\n")
        cli_outputs[session_id] = cli_output
        
        return jsonify({
            'success': True,
            'response': response,
            'cli_output': '\n'.join(cli_output)
        })

    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n🌍 TerraIntel Web Interface")
    print("----------------------------")
    print("Required files in project directory:")
    print("   - .env file with:")
    print("     GROQ_API_KEY=your_key_here")
    print("   - outputs/tactical_data.json")
    print("   - analysed_data/terrain_data.jsonl")
    print("   - outputs/map.html")
    print("\nStarting server on http://localhost:5000")
    print("----------------------------\n")
    app.run(debug=True, port=5000) 