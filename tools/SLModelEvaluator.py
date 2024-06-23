import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from IPython.display import display
from sklearn import set_config
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)


class SLModelEvaluator:
    """
    A class to evaluate and display model performance results.

    Methods
    -------
    display_results(X_valid, y_valid, best_models, best_params, best_scores, best_model_name, help_text=False):
        Displays the best parameters and evaluation metrics.
    validate_on_test(X_test, y_test, best_model, best_model_name):
        Validates the best model on the test set and displays evaluation metrics.
    visualize_pipeline(model_name, best_models):
        Visualizes the pipeline structure for a given model.
    feature_importance(X_train, y_train, df_original):
        Displays the feature importances using a RandomForest model.
    plot_roc_curve():
        Plots the ROC curve and displays the AUC score.
    plot_confusion_matrix():
        Plots the confusion matrix.
    """

    def __init__(self):
        """
        Инициализация класса SLModelEvaluator.
        """
        pass

    def display_results(
        self,
        X_valid,
        y_valid,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        help_text=False,
    ):
        """
        Displays the best parameters and evaluation metrics.

        Parameters
        ----------
        X_valid : pd.DataFrame
            Validation data.
        y_valid : pd.Series
            Target values for the validation data.
        best_models : dict
            Dictionary of best models found by grid search for each model.
        best_params : dict
            Dictionary of best hyperparameters found by grid search for each model.
        best_scores : dict
            Dictionary of best scores achieved by grid search for each model.
        best_model_name : str
            Name of the best model based on the evaluation score.
        help_text : bool, optional
            Whether to display help text explaining the metrics (default is False).
        """
        results = []
        for model_name, model in best_models.items():
            y_pred = model.predict(X_valid)
            if hasattr(model, "predict_proba"):
                score = accuracy_score(y_valid, y_pred)
                f1 = f1_score(y_valid, y_pred, average="weighted")
                precision = precision_score(y_valid, y_pred, average="weighted")
                recall = recall_score(y_valid, y_pred, average="weighted")
                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": score,
                        "F1 Score": f1,
                        "Precision": precision,
                        "Recall": recall,
                    }
                )
            else:
                mae = mean_absolute_error(y_valid, y_pred)
                mape = mean_absolute_percentage_error(y_valid, y_pred)
                r2 = r2_score(y_valid, y_pred)
                results.append(
                    {
                        "Model": model_name,
                        "R²": r2,
                        "MAE": mae,
                        "MAPE": mape,
                    }
                )

        results_df = pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
        )
        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )

        best_model_df = pd.DataFrame(
            {
                "Overall Best Model": [best_model_name],
                "Score (based on cross-validation score)": [
                    best_scores[best_model_name]
                ],
            }
        )

        print("Evaluation Metrics for Validation Set:")
        display(results_df)

        print("\nBest Parameters for Each Model (found during cross-validation):")
        display(param_df)

        print("\nOverall Best Model and Score (based on cross-validation score):")
        display(best_model_df)

        if help_text:
            print("\nMetric Explanations:")
            if hasattr(best_models[best_model_name], "predict_proba"):
                print(
                    "Accuracy: The ratio of correctly predicted instances to the total instances."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print("F1 Score: The harmonic mean of precision and recall.")
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "Precision: The ratio of correctly predicted positive observations to the total predicted positives."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "Recall: The ratio of correctly predicted positive observations to the all observations in actual class."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
            else:
                print(
                    "R²: The proportion of the variance in the dependent variable that is predictable from the independent variables."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "MAE: The average of the absolute errors between the predicted and actual values."
                )
                print("  - Range: [0, ∞), lower is better.")
                print("  - Lower values indicate better model performance.")
                print(
                    "MAPE: The mean of the absolute percentage errors between the predicted and actual values."
                )
                print("  - Range: [0, ∞), lower is better.")
                print("  - Lower values indicate better model performance.")

    def validate_on_test(self, X_test, y_test, best_model, best_model_name):
        """
        Validates the best model on the test set and displays evaluation metrics.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            Target values for the test data.
        best_model : model
            Best model found by grid search.
        best_model_name : str
            Name of the best model.
        """
        y_pred = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            evaluation_df = pd.DataFrame(
                {
                    "Accuracy": [score],
                    "F1 Score": [f1],
                    "Precision": [precision],
                    "Recall": [recall],
                },
                index=[best_model_name],
            )
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evaluation_df = pd.DataFrame(
                {"R²": [r2], "MAE": [mae], "MAPE": [f"{mape:.2%}"]},
                index=[best_model_name],
            )

        print(f"Results for {best_model_name}:")
        display(evaluation_df)

    def visualize_pipeline(self, model_name, best_models):
        """
        Visualizes the pipeline structure for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model to visualize.
        best_models : dict
            Dictionary of best models found by grid search for each model.
        """
        set_config(display="diagram")
        return best_models[model_name]

    def feature_importance(
        self,
        X_train,
        y_train,
        df_original,
        model_type="random_forest",
        print_zero_importance=False,
        importance_threshold=0.0,
    ):
        """
        Displays the feature importances using a specified ensemble model and outputs a summary table.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target values for the training data.
        df_original : pd.DataFrame
            Original dataframe with feature names.
        model_type : str, optional
            Type of model to use for feature importance ('random_forest', 'gradient_boosting').
        print_zero_importance : bool, optional
            Whether to print features with zero importance (default is False).
        importance_threshold : float, optional
            Threshold below which features are considered for listing (default is 0.0).
        """
        feature_names = df_original.columns

        # Initialize the model
        if model_type == "random_forest":
            model = (
                RandomForestClassifier(n_estimators=100, random_state=42)
                if y_train.nunique() > 2
                else RandomForestRegressor(n_estimators=100, random_state=42)
            )
        elif model_type == "gradient_boosting":
            model = (
                GradientBoostingClassifier(n_estimators=100, random_state=42)
                if y_train.nunique() > 2
                else GradientBoostingRegressor(n_estimators=100, random_state=42)
            )
        else:
            raise ValueError(
                "Unsupported model type. Choose from 'random_forest' or 'gradient_boosting'."
            )

        # Fit the model
        model.fit(X_train, y_train)

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        sorted_importances = importances[indices]
        sorted_features = feature_names[indices]

        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame(
            {"Feature": sorted_features, "Importance": sorted_importances}
        )

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance ({model_type.replace('_', ' ').title()})")
        plt.barh(range(len(indices)), sorted_importances, align="center")
        plt.yticks(range(len(indices)), sorted_features)
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.show()

        # Print features with low importance if requested
        if print_zero_importance:
            low_importance_features = importance_df[
                importance_df["Importance"] <= importance_threshold
            ]
            display(low_importance_features)

            # Summary statistics for low importance features
            summary_stats = (
                low_importance_features["Importance"].describe().to_frame().transpose()
            )
            summary_stats["sum"] = low_importance_features["Importance"].sum()
            summary_stats = summary_stats[["mean", "50%", "min", "max", "sum"]]
            summary_stats.rename(columns={"50%": "median"}, inplace=True)

            print("\nSummary Statistics for Low Importance Features:")
            display(summary_stats)

            # Prepare the list of low-importance features
            low_importance_features_list = low_importance_features["Feature"].tolist()

        return low_importance_features_list

    def plot_roc_curve(self, model, X_test, y_test):
        """
        Plots the ROC curve and displays the AUC score.

        Parameters
        ----------
        model : estimator
            The trained model.
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            True labels for test data.
        """
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("The model does not have predict_proba method.")

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, model, X_test, y_test):
        """
        Plots the confusion matrix for the model.

        Parameters
        ----------
        model : trained model
            The trained model to evaluate.
        X_test : pd.DataFrame
            The test data.
        y_test : pd.Series
            The true labels for the test data.
        """
        y_pred = model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap="Blues"
        )
        disp.ax_.set_title("Confusion Matrix")
        plt.show()
