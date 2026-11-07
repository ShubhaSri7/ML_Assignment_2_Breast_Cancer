# Breast Cancer Classification – Machine Learning Assignment 2

## Problem Statement
Build and compare five supervised machine-learning classification models to classify breast tumours as Benign or Malignant. The application evaluates a labelled test CSV using saved models; it does not retrain on uploaded test data.

## Dataset Description
- Dataset: Breast Cancer Wisconsin Diagnostic (built into scikit-learn)
- Source in code: `sklearn.datasets.load_breast_cancer(as_frame=True)`
- Instances: 569
- Features: 30 numeric diagnostic features
- Target: `Diagnosis` (`Benign` or `Malignant`)
- Split: 80% training and 20% testing, stratified, `random_state=42`
- The original dataset CSV is not required because the dataset is provided by scikit-learn. `test_data.csv` is generated from the held-out test split and is supplied for app testing.

## Models
1. Logistic Regression (with StandardScaler)
2. Decision Tree
3. K-Nearest Neighbors (with StandardScaler)
4. Gaussian Naive Bayes
5. Support Vector Machine with RBF kernel (with StandardScaler)

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC). Malignant is treated as the positive class.

## Model Performance Comparison
The measured values are in `model_comparison.csv`.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9649 | 0.9960 |      0.9750 |   0.9286 | 0.9512 | 0.9245 |
| Decision Tree          |     0.9298 | 0.9246 |      0.9048 |   0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors    |     0.9561 | 0.9823 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Naive Bayes            |     0.9386 | 0.9934 |      1.0000 |   0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |     0.9737 | 0.9947 |      1.0000 |   0.9286 | 0.9630 | 0.9442 |

## Observations
Use the actual results above to discuss the strengths and weaknesses of each model. Consider the precision/recall balance for the Malignant class, AUC, F1-score and MCC. Do not select a winner based on accuracy alone.

## Overall Best Model
Based on this run, the model with the highest F1-score for the Malignant class is **Support Vector Machine** (F1 = 0.9630). Explain whether this is also the best choice when considering AUC, recall and MCC.

## Run Locally
1. Extract this ZIP file.
2. Open a terminal in the extracted `ML_Assignment_2_Breast_Cancer` folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Select a model and upload `test_data.csv`.

## GitHub and Deployment
- GitHub repository: add your repository URL here after creating it.
- Streamlit app: add your deployed app URL here after deploying it.

## Project Files
- `app.py`: Streamlit interface and evaluation.
- `test_data.csv`: held-out test set with target labels.
- `model_comparison.csv`: model metric comparison.
- `model/`: five serialized, trained pipelines/classifiers.
- `requirements.txt`: Python dependencies.
