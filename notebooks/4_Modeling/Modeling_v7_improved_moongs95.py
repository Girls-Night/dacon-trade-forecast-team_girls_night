import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import warnings
import gc

warnings.filterwarnings("ignore")

def load_monthly_data(path: str):
    """train_month.csv를 읽어 item_id × ym pivot 생성."""
    df = pd.read_csv(path)
    df["ym"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    monthly = (
        df.groupby(["item_id", "ym"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "value_sum"})
    )
    pivot = (
        monthly.pivot_table(index="ym", columns="item_id", values="value_sum", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    return monthly, pivot

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(
    pivot: pd.DataFrame,
    max_lag: int = 6,
    min_nonzero: int = 12,
    corr_threshold: float = 0.4,
) -> pd.DataFrame:
    items = pivot.columns.tolist()
    pairs = []

    for leader in items:
        leader_series = pivot[leader].values.astype(float)
        if np.count_nonzero(leader_series) < min_nonzero:
            continue
        for follower in items:
            if leader == follower:
                continue
            follower_series = pivot[follower].values.astype(float)
            if np.count_nonzero(follower_series) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0
            # lag 1 to max_lag
            for lag in range(1, max_lag + 1):
                if len(leader_series) <= lag:
                    break
                # Leader(t-lag) vs Follower(t)
                x = leader_series[:-lag]
                y = follower_series[lag:]
                corr = safe_corr(x, y)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            if best_lag is None or abs(best_corr) < corr_threshold:
                continue

            pairs.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "best_lag": int(best_lag),
                "max_corr": float(best_corr),
            })

    return pd.DataFrame(pairs)

def _build_pair_frame(
    pivot: pd.DataFrame,
    leader: str,
    follower: str,
    best_lag: int,
    max_corr: float,
) -> pd.DataFrame:
    leader_series = pivot[leader]
    follower_series = pivot[follower]

    df = pd.DataFrame({
        "date": pivot.index,
        "b_t": follower_series.values, # Follower value at t
        "a_raw": leader_series.values, # Leader value at t
    })
    
    # --- Feature Engineering (Enhanced) ---
    
    # 1. Follower Lags & Diffs
    for lag in [1, 2, 3, 6]:
        df[f"b_t_{lag}"] = df["b_t"].shift(lag)
        
    df["b_diff1"] = df["b_t"] - df["b_t_1"]
    df["b_pct1"] = (df["b_t"] - df["b_t_1"]) / (df["b_t_1"].replace(0, np.nan) + 1e-6)
    
    # Rolling stats for Follower
    for window in [3, 6]:
        df[f"b_roll{window}_mean"] = df["b_t"].rolling(window=window, min_periods=1).mean()
        df[f"b_roll{window}_std"] = df["b_t"].rolling(window=window, min_periods=1).std()

    # 2. Leader Features (Aligned)
    # We want to predict Target = Follower(t+1).
    # Relationship: Leader(t+1 - lag) -> Follower(t+1).
    # Input available at t for predicting t+1 is Leader(t - (lag-1)).
    shift_amount = best_lag - 1
    df["a_t_aligned"] = df["a_raw"].shift(shift_amount)
    
    # Leader Lags (relative to aligned)
    for lag in [1, 2, 3]:
        df[f"a_t_aligned_{lag}"] = df["a_t_aligned"].shift(lag)
        
    df["a_diff1"] = df["a_t_aligned"] - df["a_t_aligned_1"]
    df["a_pct1"] = (df["a_t_aligned"] - df["a_t_aligned_1"]) / (df["a_t_aligned_1"].replace(0, np.nan) + 1e-6)
    
    # Rolling stats for Leader
    for window in [3, 6]:
        df[f"a_roll{window}_mean"] = df["a_t_aligned"].rolling(window=window, min_periods=1).mean()

    # 3. Interaction Features
    df["ab_ratio"] = df["a_t_aligned"] / (df["b_t"] + 1e-6)
    df["ab_diff"] = df["a_t_aligned"] - df["b_t"]

    # Target: Follower(t+1)
    df["target_value"] = df["b_t"].shift(-1)
    df["target_log1p"] = np.log1p(df["target_value"].clip(lower=0))
    df["target_date"] = df["date"] + pd.offsets.MonthBegin(1)

    # Meta
    df["leading_item_id"] = leader
    df["following_item_id"] = follower
    df["best_lag"] = best_lag
    df["max_corr"] = max_corr
    
    # Seasonality
    df["month"] = df["date"].dt.month
    
    return df

def build_training_data(pivot: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in pairs.iterrows():
        pair_frame = _build_pair_frame(
            pivot,
            leader=row["leading_item_id"],
            follower=row["following_item_id"],
            best_lag=row["best_lag"],
            max_corr=row["max_corr"],
        )
        # Drop rows with missing features (due to shifting)
        # We need at least max lag history. Max lag used in features is 6.
        pair_frame = pair_frame.dropna(subset=["b_t_6", "a_t_aligned_3"])
        frames.append(pair_frame)

    if not frames:
        raise ValueError("No training data created.")
    
    df = pd.concat(frames, ignore_index=True)
    return df

def train_cv_ensemble(train_df: pd.DataFrame, n_splits: int = 5):
    """
    Time Series Cross-Validation Ensemble.
    Splits data by date to respect time order.
    """
    # Identify unique target dates for splitting
    # We only care about dates where we have targets
    valid_dates = train_df.loc[train_df["target_log1p"].notna(), "target_date"].unique()
    valid_dates = np.sort(valid_dates)
    
    # We need at least n_splits + 1 periods to have a train set and n_splits test sets?
    # Actually sklearn TimeSeriesSplit works on indices.
    # Here we manually split by date.
    # Let's say we want to validate on the last 5 months one by one.
    
    # Strategy:
    # Fold 1: Train [:-5], Valid [-5]
    # Fold 2: Train [:-4], Valid [-4]
    # ...
    # Fold 5: Train [:-1], Valid [-1]
    
    if len(valid_dates) < n_splits + 5:
        # Fallback if not enough dates
        n_splits = 3
        
    test_dates = valid_dates[-n_splits:]
    
    models = []
    scores = []
    
    drop_cols = {"date", "target_date", "target_value", "target_log1p", "leading_item_id", "following_item_id", "a_raw"}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    
    print(f"Training with {n_splits} folds Time Series CV...")
    
    for i, test_date in enumerate(test_dates):
        train_mask = (train_df["target_date"] < test_date) & (train_df["target_log1p"].notna())
        valid_mask = (train_df["target_date"] == test_date) & (train_df["target_log1p"].notna())
        
        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "target_log1p"]
        X_valid = train_df.loc[valid_mask, feature_cols]
        y_valid = train_df.loc[valid_mask, "target_log1p"]
        
        if X_valid.empty:
            continue
            
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=2000,
            learning_rate=0.03, # Slightly lower LR for ensemble
            num_leaves=31,
            max_depth=-1,
            subsample=0.7, # More randomness for ensemble diversity
            colsample_bytree=0.7,
            random_state=42 + i, # Different seed per fold
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="mae",
            callbacks=[
                # early_stopping is handled by callbacks in new versions or just let it run
                # We'll rely on n_estimators being reasonable or manual early stopping if needed
                # For simplicity in this script, we just fit.
            ]
        )
        
        # Evaluate
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        scores.append(mae)
        models.append(model)
        
        print(f"Fold {i+1} (Test Date: {np.datetime_as_string(test_date, unit='M')}): MAE = {mae:.5f}")
        
    print(f"Average MAE: {np.mean(scores):.5f}")
    return models, feature_cols

def build_inference_features(pivot, pairs, forecast_month="2025-08"):
    target_ts = pd.Timestamp(forecast_month + "-01")
    base_ts = target_ts - pd.offsets.MonthBegin(1)
    
    rows = []
    for _, row in pairs.iterrows():
        pair_frame = _build_pair_frame(
            pivot,
            leader=row["leading_item_id"],
            follower=row["following_item_id"],
            best_lag=row["best_lag"],
            max_corr=row["max_corr"],
        )
        target_row = pair_frame[pair_frame["date"] == base_ts]
        if not target_row.empty:
            rows.append(target_row.iloc[0])
            
    return pd.DataFrame(rows)

def main():
    print("Loading data...")
    _, pivot = load_monthly_data("train_month.csv")
    
    # 1. Find pairs
    print("Finding pairs...")
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=15, corr_threshold=0.38)
    print(f"Found {len(pairs)} pairs.")
    
    # 2. Build Data
    print("Building training data...")
    train_df = build_training_data(pivot, pairs)
    
    # 3. Train Ensemble
    print("Training Ensemble Models...")
    models, feature_cols = train_cv_ensemble(train_df, n_splits=5)
    
    # 4. Inference
    print("Inference for 2025-08...")
    pred_df = build_inference_features(pivot, pairs, "2025-08")
    
    if pred_df.empty:
        print("No predictions generated!")
        return

    X_test = pred_df[feature_cols].fillna(0.0)
    
    # Ensemble Prediction
    final_preds = np.zeros(len(X_test))
    for model in models:
        final_preds += model.predict(X_test)
    final_preds /= len(models)
    
    y_pred = np.expm1(final_preds) # Inverse log1p
    
    # 5. Submission
    submission = pairs[["leading_item_id", "following_item_id"]].copy()
    submission["value"] = y_pred
    submission["value"] = submission["value"].clip(lower=0).round().astype(int)
    
    # Save
    submission.to_csv("v7_improved_submission.csv", index=False)
    print("Saved v7_improved_submission.csv")

if __name__ == "__main__":
    main()
