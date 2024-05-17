import pandas as pd
import numpy as np
from numba import jit

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



@jit(nopython=True)
def smooth_data_column(column, bonus=1000):
    N = len(column)
    costs = np.zeros(N)
    predecessors = np.zeros(N, dtype=np.int64)

    costs[0] = 0
    for i in range(1, N):
        min_cost = np.inf
        best_j = 0
        for j in range(i):
            current_cost = costs[j] + abs(column[i] - column[j]) - bonus
            if current_cost < min_cost:
                min_cost = current_cost
                best_j = j
        costs[i] = min_cost
        predecessors[i] = best_j

    path = []
    i = N - 1
    while i > 0:
        path.append(i)
        i = predecessors[i]
    path.append(0)
    path.reverse()
    smoothed_column = np.zeros(N)
    for i in range(1, len(path)):
        start = path[i - 1]
        end = path[i]
        # Linear interpolation between path points
        for j in range(start, end + 1):
            smoothed_column[j] = column[start] + (column[end] - column[start]) * ((j - start) / (end - start))

    return smoothed_column#.to_numpy()


def smooth_dataframe(dataframe):
    smoothed_data = pd.DataFrame()
    for column in dataframe.columns:
        if column != 'observedAt':
            smoothed_data[column] = smooth_data_column(dataframe[column].to_numpy())
        else:
            smoothed_data[column] = dataframe[column]
    return smoothed_data

# def smooth_dataframe(dataframe):
#     smoothed_data = dataframe.copy()
#     for column in dataframe.columns:
#         if column != 'observedAt':

#             # dataframe[f"{column}_smoothed"]=dataframe[column].rolling(4).sum()
#             smoothed_data[column]=dataframe[column].rolling(4).sum()/4
#             smoothed_data[column]=smoothed_data[column].fillna(dataframe[column])
#         else:
#             smoothed_data[column] = dataframe[column]



#     return smoothed_data


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    # data['observedAt'] = pd.to_datetime(data['observedAt'], errors='coerce')
    print(data)
    smoothed_df = smooth_dataframe(data)
    smoothed_df.index=data.index
    # smoothed_df['observedAt']=data['observedAt']

    # smoothed_df['observedAt']=data.index

    return smoothed_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'