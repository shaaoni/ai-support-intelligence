import json
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipeline.ingest import load_all_sources
from src.pipeline.quality import run_quality_checks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

def load_training_data():
    """Load and clean data for training."""
    records = load_all_sources()
    clean, _, report = run_quality_checks(records)

    # Keep only CSV tickets (markdown docs are for RAG, not classification)
    tickets = [r for r in clean if r["source"] == "csv"]
    logger.info(f"Training data: {len(tickets)} clean tickets")

    texts = [r["text"] for r in tickets]
    labels = [r["category"] for r in tickets]
    return texts, labels

def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    """Train one model, log to MLflow, return metrics."""
    with mlflow.start_run(run_name=model_name):
        # Build pipeline: TF-IDF vectorizer + classifier
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", model),
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"\n  [{model_name}]")
        print(f"    Accuracy:    {accuracy:.3f}")
        print(f"    Weighted F1: {report['weighted avg']['f1-score']:.3f}")
        print(f"\n{classification_report(y_test, y_pred)}")

        return pipeline, accuracy

def main():
    print("\n── Phase 2: ML Classifier Training ──────────────────────────")

    # Load data
    texts, labels = load_training_data()
    if len(texts) < 10:
        print("  ✗ Not enough training data. Run generate_sample_data.py first.")
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Set MLflow experiment
    mlflow.set_experiment("ticket-classifier")

    # Train all models
    best_model, best_accuracy, best_name = None, 0, ""
    for name, model in MODELS.items():
        pipeline, accuracy = train_and_evaluate(
            name, model, X_train, X_test, y_train, y_test
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name

    # Save best model
    model_path = Path("data/model")
    model_path.mkdir(exist_ok=True)
    mlflow.sklearn.save_model(best_model, path=str(model_path / "best_model"))

    # Save metadata
    meta = {"best_model": best_name, "accuracy": best_accuracy}
    with open(model_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Best model:  {best_name}")
    print(f"  Accuracy:    {best_accuracy:.3f}")
    print(f"  Saved to:    data/model/best_model")
    print(f"  ✓ Phase 2 training complete!\n")

if __name__ == "__main__":
    main()
