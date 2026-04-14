import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


def get_latest_run(model_dir: Path):
    runs = sorted(model_dir.glob("run_*"))
    if not runs:
        raise ValueError("No trained models found")
    return runs[-1]


def load_model(path):
    return joblib.load(path)


def evaluate(model, df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, pred)


def main(model_dir, test_path):
    model_dir = Path(model_dir)
    df = pd.read_csv(test_path)

    # NEW MODEL (latest run)
    latest_run = get_latest_run(model_dir)
    new_model_path = latest_run / "model.pkl"
    new_model = load_model(new_model_path)

    new_auc = evaluate(new_model, df)
    print(f"New model AUC: {new_auc:.4f}")

    # OLD MODEL (production)
    prod_model_path = model_dir / "model.pkl"

    if not prod_model_path.exists():
        print("No production model → accept new model")
        joblib.dump(new_model, prod_model_path)
        return

    old_model = load_model(prod_model_path)
    old_auc = evaluate(old_model, df)

    print(f"Old model AUC: {old_auc:.4f}")

    # 👉 DECISION
    if new_auc < old_auc:
        print("New model worse → reject")
        exit(1)

    print("✅ New model better → deploying...")

    # 👉 DEPLOY
    joblib.dump(new_model, prod_model_path)

    # OPTIONAL: update symlink
    latest_link = model_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(latest_run, target_is_directory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test", required=True)

    args = parser.parse_args()
    main(args.model_dir, args.test)