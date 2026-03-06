# Assignment 4: Containerization & Continuous Integration

This directory contains the code and configuration for Assignment 4 of the Applied Machine Learning course.

## Overview

The assignment focuses on two main DevOps concepts:
1. **Containerizing** the Flask spam classifier application (created in Assignment 3) using Docker.
2. **Continuous Integration** using a Git `pre-commit` hook to automatically run tests before allowing commits to the `main` branch.

## Files

- `app.py`: The Flask web server.
- `score.py`: The prediction logic script that loads the trained model.
- `best_spam_classifier.pkl`: The trained machine learning model from Assignment 3.
- `Dockerfile`: The instructions to build the Docker image.
- `requirements.txt`: The Python dependencies needed (matching the host environment that trained the model).
- `test.py`: A `pytest` test suite that:
  - Tests the local score logic to verify it works.
  - Builds the Docker image.
  - Runs the container on port `5000`.
  - Sends a sample POST request to the `/score` endpoint.
  - Asserts the expected output format and tears down the container.
- `coverage.txt`: The generated `pytest` coverage report demonstrating 100% coverage.
- `pre-commit`: The bash script used for the Git hook that automatically runs `test.py` on `git commit`.


