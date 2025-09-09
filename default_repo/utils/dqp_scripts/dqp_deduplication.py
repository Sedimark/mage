import sys
import json
import pandas as pd
from dqp import DataSource, DeduplicationModule


if __name__ == "__main__":
    args = sys.argv

    with open(args[1], "r") as fp:
        config = json.load(fp)
    
    module = DeduplicationModule(**config)
    
    df = pd.read_csv(args[2])
    data_source = DataSource(df)

    result = module(data_source)._df

    result.to_csv(args[2], index=False)

