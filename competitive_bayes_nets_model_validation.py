import pandas as pd
import numpy as np
from pomegranate import *
import random
from helpers import *

df = pd.read_excel("./data/competitive/games_with_cluster_by_lane.xlsx")

df = df.drop(["Unnamed: 0", "game_id"], axis=1)

df_training = df.sample(frac= 0.8 )
df_test = df.drop(df_training.index)

print("df_training: ", df_training.shape[0])
print("df_test: ", df_test.shape[0])



### takes the most repeated values
w_top_group = df_training['w_top_group'].mode()[0]
w_jungle_group = df_training['w_jungle_group'].mode()[0]
w_mid_group = df_training['w_mid_group'].mode()[0]
w_adc_group = df_training['w_adc_group'].mode()[0]
w_support_group = df_training['w_support_group'].mode()[0]

df_test['most_repeated_value'] = df_test.apply(lambda row: test_predictions(row['w_top_group'], row['w_jungle_group'], row['w_mid_group'], row['w_adc_group'], row['w_support_group'], 
                                                                       w_top_group, w_jungle_group, w_mid_group, w_adc_group, w_support_group), axis=1)



### takes random values from groups of each role
values_w_top_group = (df_training['w_top_group'].unique())
values_w_jungle_group = (df_training['w_jungle_group'].unique())
values_w_mid_group = (df_training['w_mid_group'].unique())
values_w_adc_group = (df_training['w_adc_group'].unique())
values_w_support_group = (df_training['w_support_group'].unique())

df_test['random'] = df_test.apply(lambda row: test_predictions(row['w_top_group'], row['w_jungle_group'], row['w_mid_group'], row['w_adc_group'], row['w_support_group'], 
                                                               random.choice(values_w_top_group), random.choice(values_w_jungle_group), random.choice(values_w_mid_group), random.choice(values_w_adc_group), random.choice(values_w_support_group)), axis=1)


### uses the bayes nets model
# ['chow-liu', 'exact', 'greedy', 'exact-dp']
for model_algorithm in ['chow-liu', 'exact', 'greedy']:
    try:
        model = BayesianNetwork.from_samples(df_training.to_numpy(), state_names=df_training.columns.values, algorithm=model_algorithm)
        df_test[model_algorithm] = df_test.apply(lambda row: test_model(model, row['w_top_group'], row['w_jungle_group'], row['w_mid_group'], row['w_adc_group'], row['w_support_group'], row['l_top_group'], row['l_jungle_group'], row['l_mid_group'], row['l_adc_group'], row['l_support_group']), axis=1)
        print(model_algorithm)
        model.plot(model_algorithm + ".pdf")
    except Exception as e:
        print("error: ", e)


print("most_repeated_value: ", df_test['most_repeated_value'].mean())
print("random: ", df_test['random'].mean())
for model in ['chow-liu', 'exact', 'greedy']:
    print( model, ":", df_test[model].mean())
# print(df_test['random'].value_counts())

df_test.to_excel("model_results.xlsx")




# test = model.predict([
#     [None, 0,0,0,0,1,-1,0,0,1,-1],
#     [200, 0,0,0,0,1, None,None,None,None,None],
#     [None, None,None,None,None,None, None,None,None,None,None],
# ])
# print(test)
