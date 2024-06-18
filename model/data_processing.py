import os
import json
import itertools
from transformers import AutoTokenizer

GAMES = range(1, 13)
COUNTRIES = ['AUS', 'ENG', 'FRA', 'ITA', 'RUS', 'GER', 'TUR']
MAPPING = {'AUS': 'AUSTRIA', 'ENG': 'ENGLAND', 'FRA': 'FRANCE', 'ITA': 'ITALY', 'RUS': 'RUSSIA', 'GER': 'GERMANY',
           'TUR': 'TURKEY'}
SYS_PROMPT = """You are an excellent assistant and advisor who understands and plays Diplomacy game very well. You'll be provided with board status and messages from counterpart. Please provide a two sentence suggestion whether I should trust or not trust the message from them. Your suggestion should start with these sentences: 'You should trust the message in this situation.' or 'You should not trust the message in this situation.' and then give the reason. You should consider the conversation of current and previous phases.
"""

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')


def get_turn_text(user_input, model_output, own, oppo, system_prompt=None, board_states='', cicero=None):
    system_prompt = f'<<SYS>>\n{system_prompt}\n<</SYS>>\n\n' if system_prompt is not None else ''
    cicero = f'the cicero recommended order is: {cicero}, and ' if cicero is not None else ''
    return f'<s>[INST] {system_prompt}{board_states}If {cicero}the conversation history of {own} and {oppo} are: {user_input}Should you trust {MAPPING[oppo]}? [/INST] {model_output} </s>'


def get_board_info(phase_name, board_dict):
    units = ', '.join((f'{key}: [' + ', '.join(value) + ']' for key, value in board_dict.items()))
    return f'BOARD STATES AT {phase_name}: {{{units}}}. '


def formulate_user_input(board_path, message_path, use_board=True, save_each_turn=False, only_history=False, system_prompt=False, cicero_path=None):
    data = []
    sample_id = 0
    for game in GAMES:
        with open(os.path.join(board_path, f'game{game}_board.json'), 'r', encoding='utf8') as f:
            board_infos = dict()
            for phase in json.load(f)['phases']:
                board_infos[phase['name']] = phase['init_units']
        if cicero_path is not None:
            with open(os.path.join(cicero_path, f'humangame{game}_cicero_orders.json')) as f:
                cicero_data = json.load(f)
        for own_side, counterpart in itertools.product(COUNTRIES, COUNTRIES):
            if own_side == counterpart:
                continue
            if not os.path.isfile(os.path.join(message_path, f'humangame_{game}_{own_side}_{counterpart}_result.json')):
                # print('No ' + os.path.join(message_path, f'humangame_{game}_{own_side}_{counterpart}_result.json') + ' found')
                continue
            with open(os.path.join(message_path, f'humangame_{game}_{own_side}_{counterpart}_result.json'), 'r',
                      encoding='utf8') as f:
                raw_messages = json.load(f)
            
            for phase in raw_messages:
                phase_name = phase['name']
                my_message = ''
                formulated_text = ''
                history = ''
                initial = SYS_PROMPT if system_prompt else None
                if cicero_path is not None:
                    for d in cicero_data:
                        if 'phase' in d.keys() and d['phase'] == phase_name:
                            cicero_orders = d.get('cicero_orders', {})
                            for order in cicero_orders:
                                if MAPPING[own_side] in order.keys():
                                    cicero_order = ', '.join(order[MAPPING[own_side]])
                else:
                    cicero_order = None
                for idx, message in enumerate(phase['messages']):
                    if idx == 0 and use_board:
                        my_message += get_board_info(phase_name, board_infos[phase_name])
                    if message['sender'] == MAPPING[own_side]:
                        # my_message += message['input'] + ' '
                        my_message += f"Message from {message['sender']}:'{message['message']}' "
                    if message['sender'] == MAPPING[counterpart]:
                        # user_input = my_message + ' ' + message['input']
                        user_input = my_message + f" Message from {message['sender']}:'{message['message']}'"
                        history += user_input + ' '
                        model_output = message['output']
                        if only_history:
                            formulated_text = get_turn_text(history, model_output, 
                                                            own=own_side, oppo=counterpart, 
                                                            system_prompt=initial, 
                                                            cicero=cicero_order)
                            if len(tokenizer.encode(formulated_text)) <= 4096:
                                data.append({'id': sample_id, 'text': formulated_text})
                                sample_id += 1
                        else:
                            formulated_text += get_turn_text(user_input, model_output, 
                                                             own=own_side, 
                                                             oppo=counterpart, 
                                                             system_prompt=SYS_PROMPT, 
                                                             cicero=cicero_order)
                            initial = None
                            if save_each_turn and len(tokenizer.encode(formulated_text)) <= 4096:
                                data.append({'id': sample_id, 'text': formulated_text})
                                sample_id += 1
                        my_message = ''
                if not save_each_turn and len(tokenizer.encode(formulated_text)) <= 4096 and not only_history:
                    data.append({'id': sample_id, 'text': formulated_text})
                    sample_id += 1
    return data


def generate_data(board_path, train_path, eval_path,test_path, output_file_path, 
                  use_board=True, save_each_turn=False, only_history=False, system_prompt=False, 
                  cicero_path=None):
    if cicero_path is not None:
        train_path += '_w_Cicero'
        eval_path += '_w_Cicero'
        test_path += '_w_Cicero'
    train_split = formulate_user_input(board_path, train_path, 
                                       use_board=use_board, 
                                       save_each_turn=save_each_turn, 
                                       only_history=only_history,
                                       system_prompt=system_prompt,
                                       cicero_path=cicero_path)
    eval_split = formulate_user_input(board_path, eval_path, 
                                      use_board=use_board, 
                                      save_each_turn=save_each_turn, 
                                      only_history=only_history,
                                      system_prompt=system_prompt,
                                      cicero_path=cicero_path)
    test_split = formulate_user_input(board_path, test_path, 
                                      use_board=use_board, 
                                      save_each_turn=save_each_turn, 
                                      only_history=only_history,
                                      system_prompt=system_prompt,
                                      cicero_path=cicero_path)
    with open(output_file_path, 'w') as f:
        json.dump({'train': train_split, 'eval': eval_split,'test':test_split}, f, indent=4)



if __name__ == "__main__":
    board_path = '../dataset/human_game/Board'
    train_path = '../dataset/human_game/Training'
    eval_path = '../dataset/human_game/Validation'
    test_path = '../dataset/human_game/Test'
    cicero_path = '../dataset/human_game/Cicero_orders_dataset'
    
    # generate_data(board_path, train_path, eval_path, 'no_board_history_with_sys_history_v2.json', use_board=False, only_history=True, system_prompt=True)
    generate_data(board_path, train_path, eval_path, test_path,
                  'no_board_history_with_sys_history_cicero.json', 
                  use_board=False, only_history=True, system_prompt=True, cicero_path=cicero_path)