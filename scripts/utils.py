# ml_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import shap
import lime
import lime.lime_tabular

    # -------------------
    # Exploration
    # -------------------
    def basic_info(self):
        """Print basic info about the dataset"""
        if self.data is None:
            print("Data not loaded yet!")
            return
        print("Shape:", self.data.shape)
        print("\nData types:\n", self.data.dtypes)
        print("\nMissing values:\n", self.data.isnull().sum())

    def describe(self):
        """Print statistical summary"""
        if self.data is None:
            print("Data not loaded yet!")
            return
        return self.data.describe()

    def plot_distribution(self, column):
        """Plot histogram of a column"""
        if self.data is None:
            print("Data not loaded yet!")
            return
        sns.histplot(self.data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

    def plot_correlation(self):
        """Plot correlation heatmap"""
        if self.data is None:
            print("Data not loaded yet!")
            return
        corr = self.data.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation")
        plt.show()

    # -------------------
    # Model Evaluation
    # -------------------
    def evaluate_model(self, y_true, y_pred):
        """Compute accuracy, ROC-AUC and print classification report"""
        acc = accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)
        print("Classification Report:\n", classification_report(y_true, y_pred))
        return {"accuracy": acc, "roc_auc": roc}

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    # -------------------
    # Feature Importance
    # -------------------
    def plot_feature_importance(self, model, feature_names=None, top_n=10):
        """Plot feature importance for tree-based models"""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):  # For linear models
            importance = np.abs(model.coef_[0])
        else:
            print("Model does not support feature importance")
            return

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False).head(top_n)

        sns.barplot(x="importance", y="feature", data=importance_df)
        plt.title("Top Feature Importances")
        plt.show()

    # -------------------
    # Model Interpretability
    # -------------------
    def explain_shap(self, model, X):
        """SHAP summary plot"""
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)

    def explain_lime(self, model, X_train, feature_names, instance_idx=0):
        """LIME explanation for a single instance"""
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['Not Success', 'Success'],
            mode='classification'
        )
        exp = explainer.explain_instance(
            data_row=X_train[instance_idx],
            predict_fn=model.predict_proba
        )
        exp.show_in_notebook(show_table=True)

