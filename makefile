python = venv/Scripts/python
pip = venv/Scripts/pip

setup:
	python -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

run:
	$(python) data/data_pipeline.py
	$(python) src/model_pipeline.py
mlflow:
	venv/Scripts/mlflow ui

test:
	$(python) -m pytest
  
clean:
	@if exist src\__pycache__ (rmdir /s /q src\__pycache__)
	@if exist __pycache__ (rmdir /s /q __pycache__)
	@if exist .pytest_cache (rmdir /s /q .pytest_cache)
	@if exist tests\__pycache__ (rmdir /s /q tests\__pycache__)

remove:
	@if exist venv (rmdir /s /q venv)