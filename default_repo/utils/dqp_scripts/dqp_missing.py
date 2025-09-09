from dqp import DataSource, MissingImputationModule
import pandas as pd
import sys

if __name__ == "__main__":
    args = sys.argv

    method = args[1]
    dataset_path = args[2]
    
    df = pd.read_csv(dataset_path)
    data_source = DataSource(df)
    
    config = {
        "imputation_method": method
    }

    module = MissingImputationModule(**config)

    result = module(data_source)._df

    result.to_csv(dataset_path, index=False)
