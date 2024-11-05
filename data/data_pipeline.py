import pandas as pd
import numpy as np
import logging
from sklearn.calibration import LabelEncoder

RAW_DATA_PATH = "data/raw_data.csv"
PREPROCESSED_DATA_PATH = "data/preprocessed_data.csv"

def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print("File does not exist")

def preprocess_data(df):
    encoder = LabelEncoder()
    df['left'] = df['left'].map({'yes':1,'no':0}).astype('int64')
    df['salary'] = encoder.fit_transform(df['salary'])
    df['department'] = encoder.fit_transform(df['department'])  
    return df 

def save_data(df):
    df.to_csv(PREPROCESSED_DATA_PATH, index=False)
    print("Preprocessed data save successfully!")

if __name__== "__main__":
    # Load data
    data = load_data(RAW_DATA_PATH)
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    # Save preprocessed data
    save_data(preprocessed_data)
    # log data pipeline 
    logging.basicConfig(
    filename='./logs/logger.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Data loaded, preprocessed, and saved!")