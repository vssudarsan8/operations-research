import itertools
import pandas as pd

# create a sample dataframe
df = pd.DataFrame({'product': ['A', 'B', 'C', 'D', 'E'],
                   'demand': [20, 30, 15, 40, 25]})

# generate all possible combinations of products
combinations = []
for r in range(1, len(df) + 1):
    for combo in itertools.combinations(df['product'], r):
        combinations.append(combo)

# filter combinations where the sum of demand is less than 100
valid_combinations = []
for combo in combinations:
    if df[df['product'].isin(combo)]['demand'].sum() < 100:
        valid_combinations.append(combo)

# print the valid combinations
print(valid_combinations)
