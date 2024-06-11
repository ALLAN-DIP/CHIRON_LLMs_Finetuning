from sentence_transformers import CrossEncoder
import os
import json

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
country_map = {
"AUS": "AUSTRIA", 
"ENG": "ENGLAND",
}
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
        if phase in phases:
            for order in phase["cicero_orders"]:
                if country in order or country_map[country] in order:
                    orders.append(" ".join(order[country]))
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
    for message, order in zip(messages, cicero_orders):
        label = nli.get_label(message, order)
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
        print(f"Phase")
        print(f"\tLabels: {labels}")
        print(f"\tCorrectness: {correctness}")
        print(f"\tSum Correctness: {sum(correctness)}")


if __name__ == "__main__":
    main()


