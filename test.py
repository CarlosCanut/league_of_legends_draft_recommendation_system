import pandas as pd
import numpy as np
from pomegranate import *
import random
from helpers import *

df = pd.read_excel("./model_results.xlsx")

print("most_repeated_value: ", df['most_repeated_value'].mean())
print("random: ", df['random'].mean())
for model in ['chow-liu', 'exact', 'greedy']:
    print( model, ":", df[model].mean())