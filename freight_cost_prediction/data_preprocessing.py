import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def load_data(db_path: str):
    conn= sqlite3.connect(db_path)
    df= pd.read_sql_query("select * from vendor_invoice", conn)
    conn.close()
    return df


def prepare_data(df):
    X = df[['Dollars']]
    y = df['Freight']
    return X,y


def split_data(X,y):
    return train_test_split(X, y, test_size= 0.2, random_state= 42)

















