from dqp import DataSource, DataProfilingModule
import pandas as pd
import json
import sys

if __name__ == "__main__":
    args = sys.argv

    dataset_path = args[1]
    
    df = pd.read_csv(dataset_path)
    data_source = DataSource(df)

    module = DataProfilingModule()

    result = module(data_source)
    description = result._description
    column_profiles = result._column_profiles

    print(description)
    print(column_profiles)

    # with open("/home/src/default_repo/descr.json", "w") as fp:
    #     json.dump(description, fp)

    # with open("/home/src/default_repo/profiles.json", "w") as fp:
    #     json.dump(column_profiles, fp)

    df.to_csv(dataset_path, index=False)
