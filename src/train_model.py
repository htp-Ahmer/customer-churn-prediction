"""Train, compare, and save the churn models.

Run from the repository root:
    python -m src.train_model
"""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_classifier
from src.preprocessing import build_preprocessor, load_data, split_features_target


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
RESULTS_PATH = ROOT / "models" / "model_results.json"


def main() -> None:
    df = load_data()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    metrics = {}
    for name, estimator in model_definitions.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        metrics[name] = evaluate_classifier(pipeline, x_test, y_test)

    selected_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    selected_model = trained_models[selected_model_name]
    bundle = {
        "model": selected_model,
        "model_name": selected_model_name,
        "random_state": RANDOM_STATE,
        "feature_columns": list(x.columns),
        "target_definition": "1 = Churn, 0 = Stay",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    results = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "churn_rate": float(y.mean()),
        "selected_model": selected_model_name,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved {selected_model_name} to {MODEL_PATH}")
    for name, values in metrics.items():
        print(
            f"{name}: accuracy={values['accuracy']:.3f}, "
            f"f1={values['f1_score']:.3f}, roc_auc={values['roc_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
