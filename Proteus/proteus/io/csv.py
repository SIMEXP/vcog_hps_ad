__author__ = 'Christian Dansereau'

# interact with a csv file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def write_csv(df,path):
    df.to_csv()

def read_csv(path):
    return pd.read_csv(path)

def write_xlsx(df,path):
    df.to_excel(path, sheet_name='Sheet1')

def read_xlsx(path):
    return pd.read_excel(path, 'Sheet1', index_col=None, na_values=['NA'])

