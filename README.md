import itertools
import pandas as pd


df = pd.DataFrame({'product': ['A', 'B', 'C', 'D', 'E'],
                   'demand': [20, 30, 15, 40, 25]})


combinations = []
for r in range(1, len(df) + 1):
    for combo in itertools.combinations(df['product'], r):
         
valid_combinations = []
for combo in combinations:
    if df[df['product'].isin(combo)]['demand'].sum() < 100:
        valid_combinations.append(combo)

# print the valid combinations
print(valid_combinations)
