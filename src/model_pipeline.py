from data.data_pipeline import load_data
from src.training import split_data, train_model, make_predictions, save_model
from src.evaluation import evaluate_model, log_performance, log_model
from sklearn.ensemble import RandomForestClassifier  

DATA_PATH = "data/preprocessed_data.csv"
TARGET = "left"

# Load data from CSV file
df = load_data(DATA_PATH)

# Split data 
X_train, X_test, y_train, y_test = split_data(df,TARGET)

# Train the model
rfr = RandomForestClassifier(n_estimators = 100, max_depth =10)
model = train_model(X_train, y_train, rfr)
# Predict on test data
predictions = make_predictions(X_test, model)

# Evaluate the model 
acc, f1, auc = evaluate_model(y_test, predictions)
print(f"{model} Evaluation Results:\n")
print(f"Accuracy: {auc}, F1 Score: {f1}, AUC: {auc}")

# Log performance results 
log_performance(model, y_test, predictions)

# Log the model 
log_model(model,X_test, y_test)

# Save the model 
save_model("models/{model}.pkl", model)