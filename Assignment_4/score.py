import sklearn.base
from typing import Tuple

def score(text: str, model: sklearn.base.BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Scores a text using a trained model and determines if it's spam based on a threshold.
    
    Args:
        text (str): The raw text message to classify.
        model (sklearn.base.BaseEstimator): A trained scikit-learn model or pipeline.
        threshold (float): The decision cutoff in [0, 1].
        
    Returns:
        Tuple[bool, float]: (prediction, propensity score)
    """
    # Input validation
    assert isinstance(text, str), f"Expected string for 'text', got {type(text).__name__}"
    assert isinstance(threshold, (int, float)), f"Expected numeric 'threshold', got {type(threshold).__name__}"
    assert 0.0 <= threshold <= 1.0, f"'threshold' must be between 0 and 1, got {threshold}"
    assert model is not None, "A trained model must be provided"

    # Get the probability of the positive class (Spam is usually index 1)
    propensity = float(model.predict_proba([text])[0][1])

    # Make the prediction based on the threshold
    prediction = bool(propensity >= threshold)

    return prediction, propensity