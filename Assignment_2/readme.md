Assignment 2 - DVC & MLflow

The primary focus of this assignment is 
1. establishing a robust MLOps architecture using Data Version Control (DVC) for managing data shifts and 
2. MLflow for experiment tracking, hyperparameter tuning, and model registration.

### Project Structure: 
- prepare.ipynb: Handles data ingestion, text cleaning, target encoding, and data splitting. It utilizes DVC to track different versions of the dataset and pushes the heavy CSV files to a remote Google Drive.

- train.ipynb: Defines the machine learning pipelines (TF-IDF + Classifiers), performs hyperparameter tuning via GridSearchCV, evaluates models using Area Under the Precision-Recall Curve (AUCPR), and logs all artifacts to a local MLflow tracking server.

###Workflow & Methodology: 

Part 1: Data Preparation & Versioning (prepare.ipynb)
The raw SMS data is cleaned and split into Train, Validation, and Test sets. To test model robustness against data distribution shifts, two distinct versions of the dataset were created and tracked using Git and DVC:

- Version 1: Split using random seed 21.
- Version 2: Split using random seed 77.

Part 2: Model Training & Registry (train.ipynb)
Three baseline algorithms were tested: Support Vector Machine (SVM), Logistic Regression, and Random Forest.

- To prevent data leakage, TfidfVectorizer was bundled directly with the classifiers using Scikit-Learn Pipeline objects.
- Models were tuned using GridSearchCV, optimizing specifically for AUCPR to handle the highly imbalanced nature of spam detection.
- The script automatically traverses the Git history to train models on Data Version 1, logs the results to MLflow, and then repeats the process for Data Version 2.

### Results & Champion Model
The shift in the data distribution (Seed 21 vs. Seed 77) demonstrated a clear impact on model performance, though the optimal hyperparameters remained highly robust.

- Version 1 Winner: Logistic Regression (C=5.0, solver='liblinear')
- Version 2 Winner: Support Vector Machine (C=2.0, kernel='rbf')

The Champion Model
Based on a programmatic query of the MLflow Model Registry, the Support Vector Machine trained on Version 2 Data was selected as the overall champion model.

Champion Metrics (Test Set):

- AUCPR: 0.9376
- Precision: 100.0% (0 False Positives)
- Recall: 85.71%
- F1 Score: 92.31%

In the context of a real-world spam filter, the 100% Precision score is the ideal outcome: the model successfully caught 84 spam messages without incorrectly blocking a single legitimate user message.