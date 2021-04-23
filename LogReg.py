
# Importing the libraries to be used:
import numpy as np
import pandas as pd
import sklearn

play_data = pd.read_csv("player_data.csv")
print(play_data.shape)

play_data1 = play_data.dropna(0)
print(play_data1.shape)