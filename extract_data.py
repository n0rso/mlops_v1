# Export data as a CSV for easy analysis externally

import pandas as pd
import os
import sqlite3

def export_data(filename):
    conn = sqlite3.connect(filename)
    query = "SELECT * FROM BoardGames"
    df_boardgame_full = pd.read_sql_query(query, conn)
    print(df_boardgame_full.shape)
    print(df_boardgame_full.head())

    print(list(df_boardgame_full.columns))
    return df_boardgame_full