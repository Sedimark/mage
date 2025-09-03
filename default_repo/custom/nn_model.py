import os
import numpy as np
import mlflow
import pandas as pd
# import mlflow.keras
import tensorflow as tf
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


# Set MLflow environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
os.environ['AWS_ACCESS_KEY_ID'] = 'super'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.sedimark.work'
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://mlflow.sedimark.work/")

# Create experiment
experiment_name = "NeuralNetwork_energy_consumption"
current_experiment = mlflow.get_experiment_by_name(experiment_name)

if current_experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    current_experiment = mlflow.get_experiment(experiment_id)

# End any active run
if mlflow.active_run():
    mlflow.end_run()

class NeuralNetworkModel:
    def __init__(self, input_dim, model_name='NeuralNetwork'):
        self.model_name = model_name
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mape'])

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, verbose=1):
        early_stopping = EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose,      
            callbacks=[early_stopping]
        )
        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, file_path):
        self.model.save(file_path)

    @classmethod
    def load_model(cls, file_path):
        return tf.keras.models.load_model(file_path)

# Train & Log to MLflow
@custom
def train_neural_network(data_dict, *args, **kwargs):

    X_train = pd.DataFrame.from_records(data_dict['X_train'])
    X_test = pd.DataFrame.from_records(data_dict['X_test'])
    X_val = pd.DataFrame.from_records(data_dict['X_val'])
    y_train = np.array(data_dict['y_train'])
    y_test = np.array(data_dict['y_test'])
    y_val = np.array(data_dict['y_val'])

    # print("Type and shape of X_train:", type(X_train), X_train.shape, X_train.dtype)
    # print("Type and shape of X_val:", type(X_val), X_val.shape, X_val.dtype)
    # print("Type and shape of y_train:", type(y_train), y_train.shape, y_train.dtype)
    # print("Type and shape of y_val:", type(y_val), y_val.shape, y_val.dtype)

    # print("First 5 elements of X_train:\n", X_train[:5])
    # print("First 5 elements of y_train:\n", y_train[:5])
    

    with mlflow.start_run(experiment_id=current_experiment.experiment_id):

        nn_model = NeuralNetworkModel(input_dim=X_train.shape[1])
        history = nn_model.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=16)

        # Make predictions
        y_train_pred = nn_model.predict(X_train)
        y_val_pred = nn_model.predict(X_val)

        # Compute metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = train_mse ** 0.5
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

        test_mse = mean_squared_error(y_val, y_val_pred)
        test_rmse = test_mse ** 0.5
        test_r2 = r2_score(y_val, y_val_pred)
        test_mae = mean_absolute_error(y_val, y_val_pred)
        test_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

        print(test_mse)
        print(test_rmse)
        print(test_r2)
        print(test_mae)
        print(test_mape)

        # Log metrics
        mlflow.log_metric("Train MSE", train_mse)
        mlflow.log_metric("Train RMSE", train_rmse)
        mlflow.log_metric("Train R2", train_r2)
        mlflow.log_metric("Train MAE", train_mae)
        mlflow.log_metric("Train MAPE", train_mape)

        mlflow.log_metric("Validation MSE", test_mse)
        mlflow.log_metric("Validation RMSE", test_rmse)
        mlflow.log_metric("Validation R2", test_r2)
        mlflow.log_metric("Validation MAE", test_mae)
        mlflow.log_metric("Validation MAPE", test_mape)

        # # Example input to the model
        # input_example = X_train[:5]  # 5 samples from training set

        # # Infer the signature from input and output
        # pred_example = nn_model.predict(input_example)
        # signature = infer_signature(input_example, pred_example)

        # # Save model to MLflow
        # mlflow.tensorflow.log_model(model=nn_model.model, artifact_path="model", signature=signature, 
        #                             input_example=input_example)

        # Correct shape with variable first dimension (-1)
        input_schema = Schema([
            TensorSpec(np.dtype('float32'), (-1, X_train.shape[1]), name="inputs")
        ])

        signature = ModelSignature(inputs=input_schema)

        # Prepare input example
        input_example = X_train[:5]

        # Predict using numpy values (what your model expects)
        pred_example = nn_model.predict(input_example.values)

        # Log the model
        mlflow.tensorflow.log_model(
            model=nn_model.model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )


        return {
            "metrics": {
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
                "r2": test_r2
            }
        }

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
