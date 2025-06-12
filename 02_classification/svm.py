from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os


def train_svm(X_train, X_val, X_test, y_train, y_val, y_test):
    tune = False
    if tune:
        # Random Search parameters
        random_params = {
            "C": np.logspace(-3, 3, 5),
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"] + list(np.logspace(-3, 3, 5)),
            "degree": [2, 3],
        }

        # Random Search
        random_search = RandomizedSearchCV(
            SVC(random_state=42),
            param_distributions=random_params,
            n_iter=2,
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

        # Grid Search around best parameters
        best_params = random_search.best_params_
        grid_params = {
            "C": np.logspace(
                np.log10(best_params["C"]) - 1, np.log10(best_params["C"]) + 1, 2
            ),
            "kernel": [best_params["kernel"]],
            "gamma": (
                ["scale", "auto"]
                if best_params["gamma"] in ["scale", "auto"]
                else np.logspace(
                    np.log10(best_params["gamma"]) - 1,
                    np.log10(best_params["gamma"]) + 1,
                    2,
                )
            ),
        }

        if best_params["kernel"] == "poly":
            grid_params["degree"] = [
                best_params["degree"] - 1,
                best_params["degree"],
                best_params["degree"] + 1,
            ]

        # Grid Search
        grid_search = GridSearchCV(
            SVC(random_state=42), param_grid=grid_params, cv=3, n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        best_model = random_search.best_estimator_

    # Predictions and evaluation
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        test_pred = best_model.predict(X_test)
        return {
            "model": best_model,
            "random_search_params": random_search.best_params_,
            # "grid_search_params": grid_search.best_params_,
            "train_report": classification_report(y_train, train_pred),
            "val_report": classification_report(y_val, val_pred),
            "test_report": classification_report(y_test, test_pred),
        }

    else:
        results = []

        classifier = SVC(random_state=42, probability= True)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results.append({
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist()})

        # Save your classifier or pipeline (can be e.g., a Pipeline with vectorizer + classifier)
        print(os.getcwd())
        print("Saving model...")
        joblib.dump(classifier, "text_classifier.pkl")
        print("Model saved.")

        return results
