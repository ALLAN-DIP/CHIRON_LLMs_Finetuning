import pandas as pd
def proc_df(df, list_cols):
    for col in list_cols:
        df[col] = df[col].apply(eval)
    return

def expand_df(df):
    data = []
    columns = ["message", "speaker", "receiver", "sender_label", "receiver_label", "game_id","absolute_message_index","relative_message_index"]
    for i, row in df.iterrows():
        for j in range(len(row["messages"])):
            data.append([row["messages"][j], row["speakers"][j], row["receivers"][j], row["sender_labels"][j], row["receiver_labels"][j], row["game_id"], row['absolute_message_index'], row["relative_message_index"]])
    return pd.DataFrame(data, columns = columns)

def get_dataset():
    list_cols = ["messages","sender_labels", "receiver_labels", "speakers" ,"receivers"]
    train = pd.read_csv('../dataset/intention/train.csv')
    test = pd.read_csv('../dataset/intention/test.csv')
    val = pd.read_csv('../dataset/intention/validation.csv')
    for df in [train, test, val]:
        proc_df(df, list_cols)

    train = expand_df(train)
    test = expand_df(test)
    val = expand_df(val)
    return train, test, val

def construct_hf_data(df):
    df["text"] = df["speaker"] + " sends " + df["reciever"] + ": "+df["message"]
    df["label"] = (df["sender_label"] == False) | (df["reciever_label"] == False)
    return df[['text', 'label']]

if __name__ == "__main__":
    print(f"This script should be run from the correctness folder (not root) or it will cause bugs")
    train, test, val = get_dataset()
    train = construct_hf_data(train)
    test = construct_hf_data(test)
    val = construct_hf_data(val)
    train.to_csv('../dataset/intention/train_hf.csv', index=False)
    test.to_csv('../dataset/intention/test_hf.csv', index=False)
    val.to_csv('../dataset/intention/validation_hf.csv', index=False)

