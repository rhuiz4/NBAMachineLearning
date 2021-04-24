
# Importing the libraries to be used:
import numpy as np
import pandas as pd
import sklearn


STATS_CSV = "Seasons_Stats.csv"
PLAYER_DATA_CSV = "player_data.csv"


play_data = pd.read_csv(PLAYER_DATA_CSV)
print(play_data.shape)

play_data1 = play_data.dropna(0)
print(play_data1.shape)


stats = pd.read_csv(STATS_CSV)
print(play_data.shape)

play_data1 = play_data.dropna(0)
print(play_data1.shape)