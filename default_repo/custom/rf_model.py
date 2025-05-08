import os
import mlflow
import numpy as np
import pandas as pd 
import mlflow.sklearn
import tensorflow as tf
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, log_loss

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
experiment_name = "Random_Forest_customer_churn_model"
current_experiment = mlflow.get_experiment_by_name(experiment_name)

if current_experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    current_experiment = mlflow.get_experiment(experiment_id)

# End any active run
if mlflow.active_run():
    mlflow.end_run()

@custom
def rf_model(data_dict, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    
    X_train = pd.DataFrame.from_records(data_dict['X_train'])
    X_test = pd.DataFrame.from_records(data_dict['X_test'])
    X_val = pd.DataFrame.from_records(data_dict['X_val'])
    y_train = np.array(data_dict['y_train'])
    y_test = np.array(data_dict['y_test'])
    y_val = np.array(data_dict['y_val'])


    print(current_experiment)

    with mlflow.start_run(experiment_id=current_experiment.experiment_id):

        param_grid_rf = {
            'n_estimators': [50, 100],                 # Number of trees (just one option for simplicity)
            'max_depth': [5, 10],                 # Depth of the trees (limit to two options)
            'min_samples_split': [2, 5, 10],                # Min samples to split a node (one option)
            'min_samples_leaf': [1, 2],              # Min samples to be at a leaf node (two options)
        }

        # Initialize the GridSearchCV object for Random Forest Classifier
        grid_search_rf = GridSearchCV(
            RandomForestClassifier(),
            param_grid_rf,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search_rf.fit(X_train, y_train)

        best_params_rf = grid_search_rf.best_params_

        best_model_rf = grid_search_rf.best_estimator_

        # Evaluate the best model on the test set
        y_pred = best_model_rf.predict(X_test)
        y_prob = best_model_rf.predict_proba(X_test)[:, 1]

        # Predict class probabilities and labels
        y_train_pred_proba = best_model_rf.predict_proba(X_train)[:, 1]
        y_val_pred_proba = best_model_rf.predict_proba(X_val)[:, 1]

        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

        # Training metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_log_loss = log_loss(y_train, y_train_pred_proba)

        # Validation metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_log_loss = log_loss(y_val, y_val_pred_proba)

        # Log training metrics
        mlflow.log_metric("Train Accuracy", train_accuracy)
        mlflow.log_metric("Train Precision", train_precision)
        mlflow.log_metric("Train Recall", train_recall)
        mlflow.log_metric("Train F1 Score", train_f1)
        mlflow.log_metric("Train AUC", train_auc)
        mlflow.log_metric("Train Log Loss", train_log_loss)

        # Log validation metrics
        mlflow.log_metric("Validation Accuracy", val_accuracy)
        mlflow.log_metric("Validation Precision", val_precision)
        mlflow.log_metric("Validation Recall", val_recall)
        mlflow.log_metric("Validation F1 Score", val_f1)
        mlflow.log_metric("Validation AUC", val_auc)
        mlflow.log_metric("Validation Log Loss", val_log_loss)

        # Example input to the model
        input_example = X_train[:5]  # 5 samples from training set

        # Infer the signature from input and output
        pred_example = best_model_rf.predict(input_example)
        signature = infer_signature(input_example, pred_example)


        mlflow.sklearn.log_model(
        sk_model=best_model_rf,
        artifact_path="model",
        signature=signature,
        input_example=input_example
        )

        # Generate the classification report and print it
        class_report = classification_report(y_test, y_pred, output_dict=True)

        return class_report


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
