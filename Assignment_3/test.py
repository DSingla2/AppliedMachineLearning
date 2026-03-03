import os
import time
import subprocess
import requests
import pytest
import joblib
import warnings
import numpy as np
from score import score
import app as my_app

warnings.filterwarnings("ignore")

MODEL_PATH = "best_spam_classifier.pkl"
BASE_URL = "http://127.0.0.1:5000/score"

@pytest.fixture(scope="module")
def trained_model():
    """
    Load the model just once for the whole test module using a pytest fixture.
    This saves a lot of time compared to reloading the .pkl file for every single test.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file '{MODEL_PATH}' not found. Make sure it's in the root directory.")
    return joblib.load(MODEL_PATH)



# 1. UNIT TESTS (Testing score.py directly)

def test_smoke_test(trained_model):
    """Check if the function runs without crashing and returns the expected tuple length."""
    try:
        result = score("Hello, how are you?", trained_model, 0.5)
        assert len(result) == 2
    except Exception as e:
        pytest.fail(f"score() raised an unexpected exception: {e}")

def test_format_test(trained_model):
    """Ensure the prediction and propensity outputs are the correct data types."""
    prediction, propensity = score("Sample message", trained_model, 0.5)
    assert isinstance(prediction, (bool, np.bool_, int))
    assert isinstance(propensity, float)

def test_prediction_value(trained_model):
    """Verify the prediction is only ever 0 (Ham) or 1 (Spam)."""
    prediction, _ = score("Sample message", trained_model, 0.5)
    assert int(prediction) in (0, 1)

def test_propensity_score_boundaries(trained_model):
    """Ensure the returned probabilities always stay between 0.0 and 1.0."""
    _, propensity = score("Sample message", trained_model, 0.5)
    assert 0.0 <= propensity <= 1.0

def test_threshold_zero(trained_model):
    """Edge case: If threshold is 0, everything should be flagged as Spam (1)."""
    prediction, _ = score("A completely normal ham message.", trained_model, 0.0)
    assert int(prediction) == 1

def test_threshold_one(trained_model):
    """Edge case: If threshold is 1, it should be Ham (0) unless the model is 100% certain it's spam."""
    prediction, propensity = score("URGENT! You won a million dollars!", trained_model, 1.0)
    if propensity < 1.0:
        assert int(prediction) == 0

def test_obvious_spam(trained_model):
    """Test a blatant spam message from the dataset to see if the model actually works."""
    spam_text = "ur going bahamas callfreefone 08081560665 speak live operator claim either bahamas cruise of£2000 cash 18only opt txt 07786200117"
    prediction, _ = score(spam_text, trained_model, 0.5)
    assert int(prediction) == 1

def test_obvious_non_spam(trained_model):
    """Test a standard everyday message to ensure it doesn't get false-flagged."""
    ham_text = "Hey, are we still meeting for lunch tomorrow at noon?"
    prediction, _ = score(ham_text, trained_model, 0.5)
    assert int(prediction) == 0

def test_propensity_consistent_with_threshold(trained_model):
    """Double-check that the logic separating spam and ham aligns strictly with the threshold."""
    text = "Call us now and win exciting prizes!"
    prediction_low, _ = score(text, trained_model, 0.0)
    assert int(prediction_low) == 1



# 2. INTEGRATION TESTS (Command Line / Subprocess)
@pytest.fixture(scope="module")
def flask_subprocess():
    """
    Launch the Flask app via the command line using subprocess.
    Yields to let the tests run, then safely terminates the app.
    """
    process = subprocess.Popen(["uv", "run", "python", "app.py"])
    time.sleep(5)  # Increased server sleep time to 5 seconds
    yield process
    
    # Teardown: Close the flask app using command line
    process.terminate()
    process.wait()

def test_flask_json(flask_subprocess):
    """Simulate a user sending a JSON POST request to the live server."""
    payload = {"text": "Congratulations! You won a lottery. Call now."}
    response = requests.post(BASE_URL, json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "propensity" in data

def test_flask_form_data(flask_subprocess):
    """Simulate a user sending form-data (like from an HTML form) to the live server."""
    payload = {"text": "Are we still meeting at 3pm?"}
    response = requests.post(BASE_URL, data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "propensity" in data

def test_flask_missing_text(flask_subprocess):
    """Make sure the app handles bad requests (missing text) with a 400 Bad Request error."""
    response = requests.post(BASE_URL, json={})
    assert response.status_code == 400
    assert "error" in response.json()

def test_flask_empty_text(flask_subprocess):
    """Make sure the app handles empty string payloads gracefully with a 400 error."""
    response = requests.post(BASE_URL, json={"text": "   "})
    assert response.status_code == 400
    assert "error" in response.json()


# 3. FLASK TEST CLIENT (Added to capture 100% Coverage)
# Note: pytest-cov cannot track the coverage of app.py when it is run in a separate subprocess. 
# Testing it directly via the test_client allows the coverage tool to see the execution.

@pytest.fixture
def client():
    """Create a Flask test client for internal API testing."""
    my_app.app.config['TESTING'] = True
    with my_app.app.test_client() as client:
        yield client

def test_client_prediction_form_data(client):
    """Test a valid form-data POST request using the Flask test client."""
    response = client.post("/score", data={"text": "You have won a free prize! Call now!"})
    assert response.status_code == 200

def test_client_prediction_json(client):
    """Test a valid JSON POST request using the Flask test client."""
    response = client.post("/score", json={"text": "Congratulations! You have won a lottery."})
    assert response.status_code == 200

def test_client_missing_text(client):
    """Test a JSON POST request with missing text using the Flask test client."""
    response = client.post("/score", json={})
    assert response.status_code == 400

def test_client_empty_text(client):
    """Test a JSON POST request with whitespace text using the Flask test client."""
    response = client.post("/score", json={"text": "   "})
    assert response.status_code == 400

def test_client_model_not_loaded(client, monkeypatch):
    """Simulate a server error where the model fails to load to test the 500 error block."""
    monkeypatch.setattr(my_app, "model", None)
    response = client.post("/score", json={"text": "Win a prize!"})
    assert response.status_code == 500

def test_client_internal_server_error(client, monkeypatch):
    """Simulate random code crash during scoring to test the general Exception block."""
    def mock_score_crash(*args, **kwargs):
        raise RuntimeError("Simulated crash")
    
    monkeypatch.setattr(my_app, "score", mock_score_crash)
    response = client.post("/score", json={"text": "Win a prize!"})
    assert response.status_code == 500