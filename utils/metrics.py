import numpy as np
import pandas as pd
import sklearn.metrics as skm

def speed(df: pd.DataFrame) -> dict:
    df["gt_speed"] = df["gt_speed"].fillna(0.0)
    df["est_speed"] = df["est_speed"].fillna(0.0)

    mae = skm.mean_absolute_error(df["gt_speed"], df["est_speed"])
    rmse = skm.mean_squared_error(df["gt_speed"], df["est_speed"], squared=False)
    bias = (df["est_speed"] - df["gt_speed"]).mean()

    return {"MAE": mae, 
            "RMSE": rmse, 
            "Bias": bias}

def alert(df: pd.DataFrame, speed_limit: float) -> dict:
    df["gt_speed"] = df["gt_speed"].fillna(0.0)
    df["est_speed"] = df["est_speed"].fillna(0.0)

    gt_alert = df["gt_speed"] > speed_limit
    est_alert = df["est_speed"] > speed_limit

    accuracy = skm.accuracy_score(gt_alert, est_alert)
    precision = skm.precision_score(gt_alert, est_alert, zero_division=0)
    recall = skm.recall_score(gt_alert, est_alert, zero_division=0)
    f1 = skm.f1_score(gt_alert, est_alert, zero_division=0)

    tn, fp, fn, tp = skm.confusion_matrix(gt_alert, est_alert).ravel()
    fpr = fp / (fp + tn) if (fp + tn > 0) else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False Positive Rate": fpr,
    }

def track(df: pd.DataFrame) -> dict:
    df["gt_timestamp"] = df["gt_timestamp"].fillna(0.0)
    df["start_time"] = df["start_time"].fillna(0.0)
    df["end_time"] = df["end_time"].fillna(0.0)

    duration = df["end_time"] - df["start_time"]
    coverage = ((df["start_time"] <= df["gt_timestamp"]) & (df["gt_timestamp"] <= df["end_time"])).astype(float)

    return {
        "Tracking Duration": duration.mean(),
        "Tracking Coverage": coverage.mean(),
    }

def evaluate(df: pd.DataFrame, speed_limit: float) -> dict:
    speed_metrics = speed(df)
    alert_metrics = alert(df, speed_limit)
    track_metrics = track(df)

    all_metrics = {**speed_metrics, **alert_metrics, **track_metrics}
    return all_metrics

def save(metrics: dict, path: str):
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)