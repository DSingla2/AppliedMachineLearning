# Assignment 3: Testing & Model Serving

## Overview
This repository contains a machine learning pipeline for classifying SMS/Email messages as Spam or Ham. It fulfills all requirements for Assignment 3, including a robust scoring script, a Flask API for model serving, and a comprehensive `pytest` suite that achieves 97%+ coverage.

## Repository Structure
* `best_spam_classifier.pkl`: The champion SVM model pipeline (this is the best model taken from Assignment 2)
* `score.py`: Contains the `score(text, model, threshold)` function which evaluates raw text using the trained model.
* `app.py`: A Flask web server that serves the model via a `/score` POST endpoint. (Also includes a bonus web UI at the `/` route for easy manual testing).
* `test.py`: A comprehensive test suite containing 19 tests covering unit tests, live server integration, and simulated server crashes.
* `coverage.txt`: The output report proving near 100% test coverage across the application using `pytest-cov`.
* `pytest.ini`: Configuration file for pytest (optional).
* `best_model_extract.py`: Connects to the local MLflow database that was created in Assignment 2 and extracts the best SVM_Model to use in our Flask API

---

## 1. Unit Testing (`test.py`)
The test suite utilizes `pytest` and thoroughly evaluates the `score()` function. The following test cases requested in the assignment are fully implemented:
* **Smoke test:** `test_smoke_test` ensures the function produces output without crashing.
* **Format test:** `test_format_test` validates that the output types are strictly `bool`/`int` and `float`.
* **Sanity checks:** `test_prediction_value` and `test_propensity_score_boundaries` ensure predictions are exactly 0 or 1, and propensity is bounded between 0.0 and 1.0.
* **Edge case inputs:** `test_threshold_zero` and `test_threshold_one` verify that extreme thresholds force the expected 1 (Spam) or 0 (Ham) predictions.
* **Typical inputs:** `test_obvious_spam` and `test_obvious_non_spam` verify the model accurately predicts known spam/ham strings.

## 2. Flask Serving (`app.py`)
The Flask application exposes a `/score` endpoint. 
* It accepts `POST` requests with a JSON payload (e.g., `{"text": "Win a free prize!"}`) or standard form-data.
* It returns a JSON response strictly formatted with the prediction (1 or 0) and the propensity score (float).
* *Note: Safe loading is implemented to prevent server crashes if the `.pkl` file is missing, returning a 500 status code instead.*

## 3. Integration Testing (`test.py`)
The `test_flask_...` functions handle the integration testing requirement:
* **Launch:** A `pytest` fixture uses `subprocess.Popen` to launch the Flask app via the command line in the background.
* **Test:** Python's `requests` library sends live POST requests to the `localhost:5000/score` endpoint to verify the JSON response.
* **Close:** The fixture safely issues `process.terminate()` and `process.wait()` to cleanly shut down the server after testing.

## 4. Coverage Report (`coverage.txt`)
The coverage report was generated using the following command:
```bash

uv run pytest test.py --cov=score --cov=app --cov=test --cov-report=term > coverage.txt
