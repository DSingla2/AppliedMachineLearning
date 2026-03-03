import mlflow
import joblib

def extract_champion_model():
    """
    Connects to the local MLflow database (created in Assignment 2) and extracts the best 
    SVM_Model to use in our Flask API .
    """
    try:
        # Connect to  local SQLite tracking database
        mlflow.set_tracking_uri("sqlite:///mlflow_track.db")
        
        # In notebook output, the winner was:
        # CHAMPION: SVM_Model (Version 2)
        model_name = "SVM_Model"
        
        print(f"Attempting to load '{model_name}' from MLflow...")
        
        # Load the model directly from MLflow
        model_uri = f"models:/{model_name}/2"
        best_model = mlflow.sklearn.load_model(model_uri)
        
        # Save it to the hard drive as a joblib file
        save_path = "best_spam_classifier.pkl"
        joblib.dump(best_model, save_path)
        
        print(f"Success! Champion model saved to: {save_path}")

    except Exception as e:
        print(f"Failed to extract the model. Error: {e}")

if __name__ == "__main__":
    extract_champion_model()