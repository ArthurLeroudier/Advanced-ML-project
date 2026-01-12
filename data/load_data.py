import os
import pandas as pd

def load_data()
    list_df = []
    path = os.getcwd() + "/data/csvfiles/"
    for file in os.listdir(path.replace("\\","/")):
        if file.startswith('tr'):
            list_df.append(pd.read_csv(path + file))
    df = pd.concat(list_df,axis=0)
    return df