import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from data.data_pipeline import RAW_DATA_PATH, load_data, preprocess_data 
from src.training import split_data, train_model, make_predictions

# Load data 
df =  load_data(RAW_DATA_PATH)
preprocessed_data = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(preprocessed_data,"left")
rfr = RandomForestClassifier(n_estimators = 100, max_depth = 10)

# Test data loading
def test_load_data():
    assert df.shape[0]==9540, "Dataframe size should be 9540"   
    assert df.shape[1]==10, "Dataframe columns must be 10"
    
# Test model training
def test_train_model():
    model = train_model(X_train, y_train, rfr)
    assert model is not None, "Model should be trained"
    assert hasattr(model, "estimator_"), "Model should be trained"

# Test model prediction
def test_make_predictions():
    predictions = rfr.predict(X_test)
    assert set(predictions) <= {0, 1}, "Predictions should be 0 or 1"
    
# Test cleaning data
def test_clean_data():
    assert df.isnull().sum().sum() == 0, "Dataframe must be cleaned"