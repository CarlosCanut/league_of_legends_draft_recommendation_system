import pandas as pd
import numpy as np
from pomegranate import *

df = pd.read_excel("./data/competitive/games_with_cluster_by_lane.xlsx")

df = df.drop(["Unnamed: 0", "game_id"], axis=1)
# print(df.head())

model = BayesianNetwork.from_samples(df.to_numpy(), state_names=df.columns.values, algorithm='exact')
# model.plot()

print(model.predict([
    [None, 0,0,0,0,-1,-1,0,0,1,-1],
    [200, 0,0,0,0,-1, None,None,None,None,None],
]))