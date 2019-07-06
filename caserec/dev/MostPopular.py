#%%
import os
import numpy as np
import pandas as pd 
from caserec.utils.process_data import ReadFile, ReadDataframe

#%%

train_file = './../../../../Datasets/MovieLens/100k_raw/u.data'
dict_ratings_from_file = ReadFile(train_file, sep='\t').read() 
df_ratings = pd.read_csv(train_file, sep='\t', header=None, names = ['user', 'item', 'feedback_value', 'timestamp'])
df_ratings.head()


#%%
os.listdir('./')

#%%
os.getcwd()