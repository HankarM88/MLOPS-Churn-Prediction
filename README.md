## Employee Churn Prediction MLOps Project
This project leverages an end-to-end MLOps pipeline to predict whether an employee will leave the company or not, encompassing data processing, model training, evaluation, and deployment in a reproducible and scalable manner.

### Setup and Usage
To get started with the project, follow the steps below:

#### 1. Clone the Repository
Clone the project repository from GitHub:
```bash
git clone https://github.com/HankarM88/MLOPS-for-Churn-Prediction.git
```
```bash
cd MLOps-for-Churn-Prediction
```
#### 2. Set Up the Environment
Ensure you have Python 3.8+ installed. Create a virtual environment and install the necessary dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Alternatively, you can use the makefile commands:
```bash
make setup
```
#### 3. Data Preprocessing 
load raw data from data folder, clean and preprocess it, and then store it for machine learning models:
```bash
python data/data_pipeline.py
```

#### 4. Train, Predict and Evaluate 
To train the model, run the following command:

```bash
python src/model_pipeline.py 
```
Or use the Makefile command:

```bash
make run
```
This script will load the data, preprocess it, train the model, and save the trained model to the models/ directory.

#### 5. Deployement with FastAPI
Run the FastAPI application by running:

```bash
uvicorn app:app --reload
```

#### 6. Containerization with Docker
To build the Docker image and run the container:

```bash
docker build -t churn_fastapi .
```
```bash
docker run -p 80:80 churn_fastapi
```
Once your Docker image is built, you can push it to Docker Hub, making it accessible for deployment on any cloud platform.


