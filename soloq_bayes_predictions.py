import pandas as pd
import numpy as np
from pomegranate import *
import pprint

games_by_result = pd.read_excel("./data/soloq/games_by_result_with_cluster_by_lane.xlsx")
games_by_side = pd.read_excel("./data/soloq/games_by_side_with_cluster_by_lane.xlsx")

games_by_result = games_by_result.drop(["Unnamed: 0", "game_id"], axis=1)
games_by_side = games_by_side.drop(["Unnamed: 0", "game_id"], axis=1)
# print(df.head())

model_by_result = BayesianNetwork.from_samples(games_by_result.to_numpy(), state_names=games_by_result.columns.values, algorithm='exact')
model_by_side = BayesianNetwork.from_samples(games_by_side.to_numpy(), state_names=games_by_side.columns.values, algorithm='exact')

# model_by_result.plot()

pprint.pprint(model_by_result.predict([
    [200, None,None,None,None,None,None,None,None,None,None],
    [200, -1, -1, 0, 3, 1, None,None,None,None,None],
    [None, -1, -1, 0, 3, 1, None,None,None,None,None],
    [100, None, None, None, None, None, None,1,None,2,None],
    # [100, 0,0,1,0,1, None,None,None,None,None],
    # [None, None,1,None,1,None, 0,None,1,None,0],
    # [100, None,None,None,None,None, None,None,None,None,None],
]))
print("")
pprint.pprint(model_by_side.predict([
    [200, None,None,None,None,None,None,None,None,None,None],
    [200, -1, -1, 0, 3, 1, None,None,None,None,None],
    [None, -1, -1, 0, 3, 1, None,None,None,None,None],
    [200, None, None, None, None, None, None,1,None,2,None],
    [None, -1,	-1,	2,	2,	1,	-1,	-1,	2,	3,	1],
    [None, -1,	-1,	None, None,	1,	-1,	-1,	2,	None, None],

    # [100, 0,0,1,0,1, None,None,None,None,None],
    # [None, None,1,None,1,None, 0,None,1,None,0],
    # [100, None,None,None,None,None, None,None,None,None,None],
]))

