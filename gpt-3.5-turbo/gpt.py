import os
from openai import OpenAI, AsyncOpenAI, BadRequestError
import asyncio
import time
import json
from datetime import timedelta
# OpenAI API
# import tiktoken
client = AsyncOpenAI(api_key="")
MAPPING = {'AUS': 'AUSTRIA', 'ENG': 'ENGLAND', 'FRA': 'FRANCE', 'ITA': 'ITALY', 'RUS': 'RUSSIA', 'GER': 'GERMANY',
           'TUR': 'TURKEY'}

async def create_assistant():
    assistant = await client.beta.assistants.create(
        name="Diplomacy Tutor",
        instructions="You are an excellent assistant and advisor who understands and plays Diplomacy game very well. You'll be provided with board status and messages from counterpart. Please provide a two sentence suggestion whether I should trust or not trust the message from them. Your whole suggestion should start with one of these sentences: 'You should trust the message.' or 'You should not trust the message.' and then give the reason. You should consider the conversation of current and previous phases.",
        model="gpt-3.5-turbo"
    )

    return assistant


async def add_message_to_thread(thread_id, user_question):
    # Create a message inside the thread
    message = await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content= user_question
    )
    return message


async def get_answer(assistant_id, thread):
    print("Thinking...")
    # run assistant
    print("Running assistant...")
    # run =  await client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=assistant_id
    # )

    # # wait for the run to complete
    # while True:
    #     runInfo = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    #     if runInfo.completed_at:
    #         print(f"Run completed")
    #         break

    #     print("Waiting 1sec...")
    #     time.sleep(1)

    while True:
        try:
            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
        except BadRequestError as e:
            if 'already has an active run' in str(e):
                print("Waiting for the existing run to complete...")
                await asyncio.sleep(5)
                continue
            else:
                raise

        counter = 0
        while True:
            runInfo = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if runInfo.completed_at:
                print("Run completed")
                break

            if counter >= 10:
                print("Timeout reached, restarting run...")
                break

            print("Waiting 1sec...")
            time.sleep(1)
            counter += 1

        if runInfo.completed_at:
            break

    print("All done...")
    # Get messages from the thread
    messages = await client.beta.threads.messages.list(thread.id)
    message_content = messages.data[0].content[0].text.value
    return message_content

async def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    
    async def main():
        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKCYAN = '\033[96m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'


        folder_path = "game_info"
        cicero_folder = "../dataset/human_game/Cicero_orders_dataset"

        # Get all file names under the folder
        file_names = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        flg = True
        country_mapping = {'AUS':'AUSTRIA','ENG':'ENGLAND','FRA':'FRANCE','ITA':'ITALY','RUS':'RUSSIA','GER':'GERMANY','TUR':'TURKEY'}
        for file in file_names:
            if flg:
                flg = False
                continue
            print(file)
            game_num = int(file.split('_')[1])
            cicero_file = os.path.join(cicero_folder, f'humangame{game_num}_cicero_orders.json')
            print(game_num)
            power_1 = file.split('_')[2]
            power_2 = file.split('_')[3].split('.')[0]
            power_list=[[power_1,power_2],[power_2,power_1]]
            thread = await client.beta.threads.create()
            print("Created thread with id:" , f"{bcolors.HEADER}{thread.id}{bcolors.ENDC}")
            assistant = await create_assistant()
            print(power_list)

            json_file_path=file
            with open(f'{folder_path}/{json_file_path}', 'r') as file:
                data = json.load(file)
            phases = data.get('phases', [])

            json_file_path_2="../dataset/human_game/Board/game12_board.json"
            with open(json_file_path_2, 'r') as file:
                data_2 = json.load(file)
            phases_2 = data_2.get('phases', [])
            for power_name in power_list:
                print(power_name)
                with open(cicero_file) as cf:
                    cicero_data = json.load(cf)
                for i in range(len(phases)):
                    phase_name =phases[i].get('name')

                    # state = phases_2[i].get('state')
                    # units = phases_2[i].get('init_units')
                    # phase_name_2 = phases[i].get('name')
                    for phase in phases_2:
                        if phase['name'] == phase_name:
                            m = phase.get('init_units', {})
                    cicero_order = ''
                    for d in cicero_data:
                        if 'phase' in d.keys() and d['phase'] == phase_name:
                            cicero_orders = d.get('cicero_orders', {})
                            for order in cicero_orders:
                                if MAPPING[power_name[0]] in order.keys():
                                    cicero_order = f'If the cicero recommended order for me is: ' \
                                        + ', '.join(order[MAPPING[power_name[0]]]) + f'. Should I trust {MAPPING[power_name[1]]}?'

                    question = f"Now it's {phase_name}, here is the board status of {phase_name}:{m}. I am {country_mapping[power_name[0]]}"
                    print(f"User input:{question}")
                    await add_message_to_thread(thread.id, question)
                    #tokens = await num_tokens_from_string(question, "cl100k_base")
                    #print(f"token number of current message:{tokens}")
                    #message_content = await get_answer(assistant.id, thread)
                    #print(message_content)
                    messages = phases[i].get('messages', [])
                    for message in messages:
                        print('------------------')
                        if message['sender'] ==country_mapping[power_name[0]]:
                            input = f"My response to {message['recipient']}:'{message['message']}'."
                            print(f"User input:{input}")
                            await add_message_to_thread(thread.id, input)
                            message_content = ''
                            print(f"Gpt output:{message_content}")

                        if message['sender'] ==country_mapping[power_name[1]]:
                            input = f"Message from {message['sender']}:'{message['message']}'." + cicero_order
                            print(f"User input:{input}")

                            await add_message_to_thread(thread.id, input)
                            
                            message_content = await get_answer(assistant.id, thread)
                            print(f"Gpt output:{message_content}")
                        
                        message['input'] = input
                        # question = input
                        # print(question)
                        # await add_message_to_thread(thread.id, question)
                        # message_content = await get_answer(assistant.id, thread)
                        # print(f"FYI, your thread is: , {bcolors.HEADER}{thread.id}{bcolors.ENDC}")
                        # print(f"FYI, your assistant is: , {bcolors.HEADER}{assistant.id}{bcolors.ENDC}")
                        message['output'] = message_content
                        
                    # question = "What's the current phase? When providing the last suggestion, What phases board status did you use, tell me the phase name? Did you consider the previous conversation history before the current phase?"
                    # print(question)
                    # await add_message_to_thread(thread.id, question)
                    # message_content = await get_answer(assistant.id, thread)
                    # print(message_content)

                output_file_path = f"NEW_FILES/humangame_{game_num}_{power_name[0]}_{power_name[1]}_result.json"
                with open(output_file_path, 'w') as output_file_3:
                    json.dump(phases, output_file_3, indent=2)
                print(f"whole_phases processed and saved to {output_file_path}")



    start_time = time.time()

    asyncio.run(main())

    print(str(timedelta(seconds=time.time() - start_time)))