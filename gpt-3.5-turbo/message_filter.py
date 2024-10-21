import json

def reformat_data(data):
    phases = {}
    for i in range(len(data['messages'])):
        phase_name = data['seasons'][i][0] + data['years'][i] + 'M'
        message = {
            "sender": data['speakers'][i].upper(),
            "recipient": data['receivers'][i].upper(),
            "phase": phase_name,
            "message": data['messages'][i]
        }
        if phase_name not in phases:
            phases[phase_name] = {"name": phase_name, "messages": []}
        phases[phase_name]["messages"].append(message)
    return {"phases": list(phases.values())}

# Load original data


with open('test.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        players = data['players']
        gameID = data['game_id']
        reformatted_data = reformat_data(data)

# Save reformatted data into another file
        with open(f'game_info/humangame_{gameID}_{players[0].upper()[:3]}_{players[1].upper()[:3]}.json', 'w') as file:
            json.dump(reformatted_data, file, indent=2)
