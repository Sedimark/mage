import pandas as pd
import os
import pickle
import uuid
from minio import Minio
from io import BytesIO
from default_repo.utils.mlflow_inference.default import load_and_predict
from crossformer.utils.tools import Preprocessor

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def send_result_to_minio(file_name: str, predictions: pd.DataFrame):
    try:
        minioClient = Minio(
            os.getenv("MLFLOW_S3_ENDPOINT_URL", "").split("/")[-1],
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False)

        csv_bytes = predictions.to_csv(index=True).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        minioClient.put_object(
            "predictions",
            file_name,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type='application/csv')
    except Exception as e:
        print(e)

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
    model_name, value_cols, in_len, data = data
    prediction_uuid = uuid.uuid4()
    prediction_name = f'{kwargs.get("prediction_name", "prediction")}_{prediction_uuid}.csv'
    
    preprocessor = Preprocessor(method="zscore",per_feature=True)
    preprocessor.fit(data[value_cols].values)
    data[value_cols] = preprocessor.transform(data[value_cols].values)
    stats = preprocessor.export()

    with open("default_repo/scaler_config.pkl", "wb") as f:
        pickle.dump(stats, f)

    result = load_and_predict(
        model_name=model_name.split("/")[1],
        model_version=model_name.split("/")[-1],
        data=data[value_cols].iloc[-in_len:],
        validate_output=True
    )

    if not result["validation_passed"] or result["predictions"] is None:
        return data.to_dict("records")

    predictions = result["predictions"]

    send_result_to_minio(prediction_name, predictions)

    predictions.columns = value_cols
    interval = data['observedAt'].iloc[-1] - data['observedAt'].iloc[-2]
    predictions['observedAt'] = interval
    predictions['observedAt'] = predictions['observedAt'].cumsum() + data['observedAt'].iloc[-1]

    datasetId_cols = [col.replace('__value', '__datasetId') for col in value_cols]
    predictions[datasetId_cols] = f'urn:ngsi-ld:Prediction:{prediction_uuid}'

    type_cols = [col.replace('__value', '__type') for col in value_cols]
    predictions[type_cols] = 'Property'

    # data = pd.concat([data, predictions], ignore_index=True)
    data = predictions
    print(data.tail(20))

    return data.to_dict("records")


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'