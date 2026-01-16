import os
import pandas as pd
import numpy as np

def load_data()
    list_df = []
    path = os.getcwd() + "/data/csvfiles/"
    t = 0 #week number
    for file in os.listdir(path.replace("\\","/")):
        if file.startswith('tr'):
            new_df = pd.read_csv(path + file)[['SHOP_WEEK', 'QUANTITY', 'SPEND', 'PROD_CODE', 'STORE_CODE']]
            new_df[['SHOP_WEEK']] = np.ones((1,len(new_df[['SHOP_WEEK']])))
            list_df.append(new_df)
            t+=1
    df = pd.concat(list_df,axis=0)
    df = df[df['QUANTITY'] > 0].copy()

    #On ne prend que les magasins avec le plus de transactions
    nb_stores = 10
    top_stores = df['STORE_CODE'].value_counts().head(nb_stores).index.tolist()
    df_top = df[df['STORE_CODE'].isin(top_stores)]

    nb_products = 50
    top_products = df_top['PROD_CODE'].value_counts().head(nb_products).index.tolist()
    df_top = df_top[df_top['PROD_CODE'].isin(top_products)]

    #On prend le prix unitaire par transaction
    df_top['UNIT_PRICE'] = df_top['SPEND'] / df_top['QUANTITY'] 

    df_final = df_top.pivot_table(
        index=['SHOP_WEEK', 'STORE_CODE'],
        columns='PROD_CODE',
        values='UNIT_PRICE',
        aggfunc='mean'
    )
    df_final.reset_index(inplace=True)

    #fill the NANs with forward method
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    return df_final