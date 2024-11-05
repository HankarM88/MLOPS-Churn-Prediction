import pickle
from sklearn.model_selection import train_test_split 
 
# Split data
def split_data(df, target, test_size=0.2):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Train ML model 
def train_model(X_train, y_train, model):
    print(f"Training {model}...")
    model.fit(X_train, y_train)
    return model

# Predict on X_test
def make_predictions(X_test, model):
    predictions = model.predict(X_test)
    return predictions

# Save model
def save_model(path, model):
    with open(path,'wb') as file:
        pickle.dump(model,file)
        print(f"Model stored in {path}")

