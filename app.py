import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🔬", layout="wide")
st.title("Breast Cancer Classification")
st.write("Upload a labelled test CSV and evaluate one of the five saved classification models.")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

selected_name = st.selectbox("Select a trained model", list(MODEL_FILES))
uploaded_file = st.file_uploader("Upload test_data.csv (must include Diagnosis)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded data")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        if "Diagnosis" not in df.columns:
            st.error("The CSV must contain a target column named 'Diagnosis'.")
            st.stop()

        y_test = df["Diagnosis"]
        X_test = df.drop(columns=["Diagnosis"])

        model_path = MODEL_DIR / MODEL_FILES[selected_name]
        if not model_path.exists():
            st.error(f"Saved model not found: {model_path}")
            st.stop()

        model = load_model(str(model_path))
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X_test.columns]
        extra = [c for c in X_test.columns if c not in expected]
        if missing or extra:
            if missing:
                st.error(f"Missing feature columns: {missing}")
            if extra:
                st.warning(f"Unexpected columns (will be ignored): {extra}")
            if missing:
                st.stop()

        X_test = X_test[expected]
        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error("Please remove missing values before evaluation.")
            st.stop()

        allowed = set(model.classes_)
        unknown = set(y_test.unique()) - allowed
        if unknown:
            st.error(f"Unknown target labels: {sorted(unknown)}. Expected: {sorted(allowed)}")
            st.stop()

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="Malignant", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="Malignant", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="Malignant", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        auc = None
        if hasattr(model, "predict_proba") and "Malignant" in model.classes_:
            probabilities = model.predict_proba(X_test)
            malignant_index = list(model.classes_).index("Malignant")
            y_binary = (y_test == "Malignant").astype(int)
            if y_binary.nunique() == 2:
                auc = roc_auc_score(y_binary, probabilities[:, malignant_index])

        st.subheader(f"Evaluation: {selected_name}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Accuracy", f"{accuracy:.4f}")
        metric_cols[1].metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
        metric_cols[2].metric("Precision", f"{precision:.4f}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Recall", f"{recall:.4f}")
        metric_cols[1].metric("F1-score", f"{f1:.4f}")
        metric_cols[2].metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion matrix")
        labels = ["Benign", "Malignant"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(
            y_test, y_pred, labels=labels, zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Predictions")
        output = X_test.copy()
        output["Actual Diagnosis"] = y_test.values
        output["Predicted Diagnosis"] = y_pred
        st.dataframe(output)
        st.download_button(
            "Download predictions CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
else:
    st.info("Choose a model and upload the supplied test_data.csv to view evaluation results.")

st.divider()
st.caption("Models are loaded from the model/ directory. The app does not retrain models on uploaded data.")
