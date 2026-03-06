import os
import subprocess
import time
import requests
import pytest

# Inject Docker bin path into PATH so docker and its credential helpers can be found
docker_bin = r"C:\Program Files\Docker\Docker\resources\bin"
if docker_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = docker_bin + os.pathsep + os.environ.get("PATH", "")

def test_local_score():
    """
    Test the local score.py logic directly before Docker is built.
    This ensures that broken logic gets blocked by pytest immediately.
    """
    from score import score
    import joblib
    
    # Load model and run score
    model = joblib.load("best_spam_classifier.pkl")
    # Using specific test that should trigger Spam (True)
    text = "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot!"
    prediction, propensity = score(text, model, 0.5)
    
    # Assert it correctly classified it
    assert prediction is True, f"Expected prediction to be True (Spam), got {prediction} with propensity {propensity}"

def test_docker():
    """
    Test to check if the Docker container launches correctly,
    responds to the /score endpoint, and shuts down properly.
    """
    # 1. Build the docker image
    build_cmd = ["docker", "build", "-t", "flask_spam_app", "."]
    print("Building Docker image...")
    subprocess.run(build_cmd, check=True)

    # 2. Run the docker container
    # Run it detached (-d) so it doesn't block the test script
    run_cmd = ["docker", "run", "-d", "-p", "5000:5000", "--name", "flask_spam_container", "flask_spam_app"]
    print("Running Docker container...")
    
    # Ensure there isn't a dangling container from a previous failed run
    subprocess.run(["docker", "rm", "-f", "flask_spam_container"], stderr=subprocess.DEVNULL)
    
    container_id = subprocess.check_output(run_cmd).decode("utf-8").strip()

    try:
        # Give the flask server time to start up inside the container
        print("Waiting for server to start...")
        time.sleep(5)

        # 3. Send a request to the localhost endpoint /score
        url = "http://localhost:5000/score"
        sample_data = {"text": "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot!"}
        
        response = requests.post(url, json=sample_data)
        
        # 4. Check if the response is as expected
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        
        json_data = response.json()
        assert "prediction" in json_data, "Response missing 'prediction' key"
        assert "propensity" in json_data, "Response missing 'propensity' key"

        print(f"Prediction: {json_data['prediction']}, Propensity: {json_data['propensity']}")
        
    finally:
        # 5. Close the docker container
        print("Stopping and removing the Docker container...")
        subprocess.run(["docker", "stop", container_id], check=True)
        subprocess.run(["docker", "rm", container_id], check=True)
