import pandas as pd


for dataset in ["cadec", "smm4h"]:
    for i in [2]:
        for mod in ["train", "test"]:
            path = f"data/{dataset}/run_{i}/{mod}.csv"
            df = pd.read_csv(path)
            df = df.drop(columns="text")
            df.to_csv(path)