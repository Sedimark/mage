if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    print(data)
    iris = datasets.load_iris()
    colors = ['red', 'green', 'blue']

    standardized_df = pd.DataFrame(data, columns=iris['feature_names'])

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(standardized_df['sepal length (cm)'], standardized_df['sepal width (cm)'], color='blue')
    plt.xlabel('Standardized Sepal Length')
    plt.ylabel('Standardized Sepal Width')
    plt.title('Standardized Iris Dataset: Sepal Length vs Sepal Width')
    plt.grid(True)
    plt.show() 
