from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 
from src.training import make_predictions 
import mlflow 
from mlflow import sklearn 
import logging

# Evaluate moe performance 
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return acc, f1, auc


# Log performance results in the logger 
def log_performance(model, y_true, y_pred):
    acc, f1, auc = evaluate_model(y_true, y_pred)
    logging.basicConfig(
    filename='logs/logger.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Model: {model}, Accuracy: {acc}, F1 Score: {f1}, AUC Score: {auc}")


# Log the model using MLflow
def log_model(model, X_test, y_test):
    with mlflow.start_run():
        preds = make_predictions(X_test, model) 
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1 score", f1)
        mlflow.sklearn.log_model(model, "random_forest_model")
        print(f"Logged model with accuracy: {acc}, and F1 score:{f1}")