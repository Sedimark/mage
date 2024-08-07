# Variables {"target_column": {"type":"str","description":"The target column for making predictions using K-Nearest Neighbors Classifier.","regex":"^.*$"}}

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transformer(data, *args, **kwargs):
    target_column = None if kwargs.get("target_column") is None else kwargs.get("target_column")
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    data['predictions'] = model.predict(X_test)
    return data

@test
def test(output, *args) -> None:
    assert output is not None, 'The output is undefined'
