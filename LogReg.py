
# Importing the libraries to be used:
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


STATS_CSV = "Seasons_Stats.csv"
PLAYER_DATA_CSV = "player_data.csv"

DEBUG = False
VIEW_SAMPLE = 10

# play_data = pd.read_csv(PLAYER_DATA_CSV)
# print(play_data.shape)

# play_data1 = play_data.dropna(0)
# print(play_data1.shape)


#Reads data, drops useless columns
stats = pd.read_csv(STATS_CSV)
stats = stats.drop(['blanl', 'blank2', 'Unnamed: 0', 'Player', 'Tm', 'GS', 'G', 'OWS', 'DWS', 'WS', 'WS/48'], axis=1)

#Keep columns with data we want, drop useless rows
stats = stats[['Pos', 'Age', 'TS%', '3PAr', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TRB%', 'USG%', 'FG%', '3P%', '2P%', 'FT%']]
stats = stats.dropna(0)
pos = ['C', 'PF', 'PG', 'SG', 'SF']
stats = stats[stats['Pos'].isin(pos)].sample(n=5000, random_state=21)

#Get and normalize the features
X = stats.drop(['Pos'], axis=1).to_numpy()
normalize(X, copy=False)

if DEBUG:
    print(X.shape)

#Get and normalize the target
y = stats['Pos']
for i in range(5):
    y = y.replace(to_replace=pos[i], value=i)
y = y.to_numpy()

if DEBUG:  
    print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

print(X_train.shape, ' - ', y_train.shape)
print(X_test.shape, ' - ', y_test.shape)

model = LogisticRegression(penalty='none', multi_class='multinomial', max_iter=10000)
model.fit(X,y)
acc = model.score(X_test, y_test)
print(acc, '%', sep='')

pred = model.predict(X_test[0:VIEW_SAMPLE])

for i in range(VIEW_SAMPLE):
    print('Predicted:', pred[i], '\tActual:', y_test[i])

coefs = model.coef_
print(coefs)