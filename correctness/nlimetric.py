from sentence_transformers import CrossEncoder
import os
import json

country_map = {
"AUS": "AUSTRIA", 
"ENG": "ENGLAND",
"FRA": "FRANCE",
"GER": "GERMANY",
"ITA": "ITALY",
"RUS": "RUSSIA",
"TUR": "TURKEY"
}

# Diplomacy Game Regions: ADR, AEG, ALB, ANK, APU, ARM, BAL, BAR, BEL, BER, BLA, BOH, BRE, BUD, BUL, BUR, CLY, CON, DEN, EAS, ECH, EDI, FIN, GAL, GAS, GOB, GOL, GRE, HEL, HOL, ION, IRI, KIE, LON, LVN, LVP, MAO, MAR, MOS, MUN, NAF, NAO, NAP, NTH, NWG, NWY, PAR, PIC, PIE, POR, PRU, ROM, RUH, RUM, SER, SEV, SIL, SKA, SMY, SPA, STP, SWE, SYR, TRI, TUN, TUS, TYR, TYS, UKR, VEN, VIE, WAL, WAR, WES, YOR
region_map = {
"ADR": "Adriatic Sea",
"AEG": "Aegean Sea",
"ALB": "Albania",
"ANK": "Ankara",
"APU": "Apulia",
"ARM": "Armenia",
"BAL": "Baltic Sea",
"BAR": "Barents Sea",
"BEL": "Belgium",
"BER": "Berlin",
"BLA": "Black Sea",
"BOH": "Bohemia",
"BRE": "Brest",
"BUD": "Budapest",
"BUL": "Bulgaria",
"BUR": "Burgundy",
"CLY": "Clyde",
"CON": "Constantinople",
"DEN": "Denmark",
"EAS": "Eastern Mediterranean",
"ECH": "English Channel",
"EDI": "Edinburgh",
"ENG": "English Channel",
"FIN": "Finland",
"GAL": "Galicia",
"GAS": "Gascony",
"GOB": "Gulf of Bothnia",
"GOL": "Gulf of Lyon",
"GRE": "Greece",
"HEL": "Helgoland Bight",
"HOL": "Holland",
"ION": "Ionian Sea",
"IRI": "Irish Sea",
"KIE": "Kiel",
"LON": "London",
"LVN": "Livonia",
"LVP": "Liverpool",
"MAO": "Mid-Atlantic Ocean",
"MAR": "Marseilles",
"MOS": "Moscow",
"MUN": "Munich",
"NAF": "North Africa",
"NAO": "North Atlantic Ocean",
"NAP": "Naples",
"NTH": "North Sea",
"NWG": "Norwegian Sea",
"NWY": "Norway",
"PAR": "Paris",
"PIC": "Picardy",
"PIE": "Piedmont",
"POR": "Portugal",
"PRU": "Prussia",
"ROM": "Rome",
"RUH": "Ruhr",
"RUM": "Rumania",
"SER": "Serbia",
"SEV": "Sevastopol",
"SIL": "Silesia",
"SKA": "Skagerrak",
"SMY": "Smyrna",
"SPA": "Spain",
"SPA/SC": "Spain/SC",
"STP": "St. Petersburg",
"SWE": "Sweden",
"SYR": "Syria",
"TRI": "Trieste",
"TUN": "Tunis",
"TUS": "Tuscany",
"TYR": "Tyrolia",
"TYS": "Tyrrhenian Sea",
"UKR": "Ukraine",
"VEN": "Venice",
"VIE": "Vienna",
"WAL": "Wales",
"WAR": "Warsaw",
"WES": "Western Mediterranean",
"YOR": "Yorkshire"
}

def get_region(region_key, verbose=False):
    to_ret = region_map.get(region_key, "UNKNOWN")
    if verbose:
        if to_ret == "UNKNOWN":
            print(f"Unknown region {region_key}")
    return to_ret

def move_mapper(move):
    parsed = move.split(" ") # Splits into unit type, loc1, action, conditional on action
    unit_type = None
    if parsed[0] == "A":
        unit_type = "army"
    elif parsed[0] == "F":
        unit_type = "fleet"
    else:
        raise ValueError(f"Invalid unit type {move}")
    if len(parsed) == 3: # then either H, B or D
        if parsed[2] == "B":
            return f"builds {unit_type} in {get_region(parsed[1])}"
        elif parsed[2] == "D":
            return f"disbands {unit_type} in {get_region(parsed[1])}"
        elif parsed[2] == "H":
            return f"holds {unit_type} in {get_region(parsed[1])}"
    elif len(parsed) == 4: # then must be move
        # rewrite w get_region
        return f"moves {unit_type} from {get_region(parsed[1])} to {get_region(parsed[3])}"
    elif len(parsed) == 5: # then must be support or recieve convoy
        if parsed[-1] == "VIA":
            return None# Since this is implied by another message which also have info on who is convoying so skipping
        secondary_unit_type = None
        if parsed[3] == "A":
            secondary_unit_type = "army"
        elif parsed[3] == "F":
            secondary_unit_type = "fleet"
        else:
            raise ValueError(f"Invalid unit type {move}")
        return f"Uses {unit_type} from {get_region(parsed[1])} to support {secondary_unit_type} in {get_region(parsed[4])}"
    elif len(parsed) == 7: # then must be convoy or support move
        secondary_unit_type = None
        if parsed[3] == "A":
            secondary_unit_type = "army"
        elif parsed[3] == "F":
            secondary_unit_type = "fleet"
        else:
            raise ValueError(f"Invalid unit type {move}")
        # use get_region
        secondary_move = f"{secondary_unit_type} in their move from {get_region(parsed[-3])} to {get_region(parsed[-1])}"
        if parsed[2] == "C":
            return f"{unit_type} in {get_region(parsed[1])} convoys {secondary_move}"
        elif parsed[2] == "S":
            return f"{unit_type} in {get_region(parsed[1])} supports {secondary_move}"
        else:
            raise ValueError(f"Invalid move format {move}")
    else:
        raise ValueError(f"Invalid move format {move}")
    return None

def report_move(country, moves, tense="future"):
    playword = "plays" if tense == "present" else "will play"
    moves_str = f"Country {country_map.get(country, 'UNKNOWN')} {playword}: "
    for move in moves:
        move_str = move_mapper(move)
        if move_str is not None:
            moves_str += move_str + ", "
    return moves_str[:-1]+"." # comma at the end is not needed

class NLIScore:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/nli-roberta-base')
        self.label_mapping = ['contradiction', 'entailment', 'neutral']

    def get_label(self, sentence1, sentence2):
        scores = self.model.predict([(sentence1, sentence2)])
        return self.label_mapping[scores.argmax(axis=1)[0]]
    

def reduce_outputs(outputs):
    # given a list of model outputs in the form {You should not trust ..../ You should not trust} 
    decisions = []
    for i in range(len(outputs)):
        decision = outputs[i].split("trust")[0]
        if "not" in decision:
            decisions.append(False)
        else:
            decisions.append(True)
    return decisions
# we want to extract it such that for each phase, we have a list of messages sent by country and the list of outputs of those messages

def get_messages_in_phase(message_data, country):
    all_messages = []
    all_outputs = []
    all_phases = []
    for phase in message_data:
        all_phases.append(phase["name"])
        messages_in_phase = []
        outputs_in_phase = []
        for message in phase["messages"]:
            if message["sender"] == country_map[country]:
                messages_in_phase.append(message["message"])
                outputs_in_phase.append(message["output"])
        all_messages.append(messages_in_phase)
        all_outputs.append(reduce_outputs(outputs_in_phase))
    return all_phases, all_messages, all_outputs

def get_cicero_orders_in_phase(phases, cicero_data, country):
    orders = []
    for phase in cicero_data:
        if len(phase) == 0:
            continue
        if phase['phase'] in phases:
            for order in phase["cicero_orders"]:
                if country_map[country] in order:
                    orders.append(report_move(country, order[country_map[country]]))
    return orders
    
def get_data(game_number=1, country_1="ENG", country_2="AUS"):
    # Get Cicero data
    with open(f"dataset/human_game/Cicero_orders_dataset/humangame{game_number}_cicero_orders.json") as f:
        cicero_data = json.load(f)
    # Get message data
    with open(f"dataset/human_game/Training/humangame_{game_number}_{country_1}_{country_2}_result.json") as f:
        message_data = json.load(f)
    phases, messages_all_phases, outputs_all_phases = get_messages_in_phase(message_data, country_2)
    cicero_orders_all_phases = get_cicero_orders_in_phase(phases, cicero_data, country_2)
    return phases, messages_all_phases, outputs_all_phases, cicero_orders_all_phases

def get_nli_score(messages, cicero_orders):
    nli = NLIScore()
    labels = []
    for message in messages:
        label = nli.get_label(message, cicero_orders)
        labels.append(label)
    return labels

# we compare labels and outputs, if labels is entails then output should be true, if labels is contradicts then output should be false, if neutral ignore
def judge_correctness(labels, outputs):
    correctness = []
    for label, output in zip(labels, outputs):
        if label == "entailment":
            correctness.append(int(output))
        elif label == "contradiction":
            correctness.append(-int(output))
        else:
            correctness.append(0)
    return correctness

def main():
    # first load game 1 with country_1 as AUS and country_2 as ENG
    game_number = 1
    country_1 = "AUS"
    country_2 = "ENG"
    phases, messages_all_phases, outputs_all_phases, cicero_orders_all_phases = get_data(game_number, country_1, country_2)
    for messages, outputs, cicero_orders in zip(messages_all_phases, outputs_all_phases, cicero_orders_all_phases):
        labels = get_nli_score(messages, cicero_orders)
        correctness = judge_correctness(labels, outputs)
        print(f"Phase:")
        for message, output, label, correct in zip(messages, outputs, labels, correctness):
            print(f"\tMessage: {message}")
            print(f"\tCicero Order: {cicero_orders}")
            print(f"\tLabel: {label}")
            print(f"\tOutput: {output}")
            print(f"\tCorrectness: {correct}")
            print()
        print(f"\tSum Correctness: {sum(correctness)}")

if __name__ == "__main__":
    main()


