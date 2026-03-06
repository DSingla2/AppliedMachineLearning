from flask import Flask, request, jsonify
import joblib
import os
import warnings
from score import score

# Ignoring warnings to keep the server console clean
warnings.filterwarnings("ignore")

app = Flask(__name__)

# I'm defining the model path here. 
MODEL_PATH = "best_spam_classifier.pkl"
model = None

# I'm checking if the file exists before loading to prevent the app from crashing on startup.
# This makes the API much safer and more robust.
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    """
    I added this simple HTML homepage so users can easily test the classifier 
    manually in their browser, rather than relying solely on backend API calls.
    """
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spam Classifier</title>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form action="/score" method="post">
                <label for="text">Enter Text:</label><br>
                <textarea id="text" name="text" rows="4" cols="50" required></textarea><br><br>
                <input type="submit" value="Check Spam">
            </form>
        </body>
        </html>
    """


@app.route("/score", methods=["POST"])
def score_endpoint():
    """
    This is the main API endpoint required by the assignment.
    It receives a text as a POST request and gives a response in JSON format 
    consisting of the prediction and propensity.
    """
    # If the model didn't load properly earlier, I catch it here and return a safe 500 error.
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        # I am handling both JSON (for automated tests/API calls) and form-data (for the HTML webpage)
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip() if data else ""
        else:
            text = request.form.get("text", "").strip()

        # Basic input validation: If the user sends empty text, return a 400 Bad Request
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Calling my scoring function from score.py
        prediction, propensity = score(text, model, 0.5)

        # Returning the exact JSON structure requested by the assignment
        return jsonify({
            "prediction": int(prediction),  # Casting to int so it returns 1 (Spam) or 0 (Ham) for my tests
            "propensity": float(propensity)
        }), 200

    except Exception as e:
        # Catch-all for any unexpected crashes to keep the server alive
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__": # pragma: no cover
    # I set use_reloader=False so it doesn't double-load the model and plays nicely with the pytest subprocess
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)