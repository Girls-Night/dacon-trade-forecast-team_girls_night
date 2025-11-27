import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import warnings

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
                # x = leader[:-lag], y = follower[lag:]
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
    
    # Follower features
    df["b_t_1"] = df["b_t"].shift(1)
    df["b_t_2"] = df["b_t"].shift(2)
    df["b_diff1"] = df["b_t"] - df["b_t_1"]
    df["b_pct1"] = (df["b_t"] - df["b_t_1"]) / (df["b_t_1"].replace(0, np.nan) + 1e-6)
    df["b_roll3"] = df["b_t"].rolling(window=3, min_periods=1).mean()
    df["b_std3"] = df["b_t"].rolling(window=3, min_periods=1).std()
    
    # Leader features (Aligned)
    # We want to predict Target = Follower(t+1).
    # Relationship: Leader(t+1 - lag) -> Follower(t+1).
    # At row `t`, we want features representing the input for predicting `t+1`.
    # The input from leader should be Leader(t+1 - lag).
    # Since lag >= 1, t+1-lag <= t.
    # So we can access Leader at index t - (lag - 1).
    # shift_amount = lag - 1.
    # If lag=1, shift=0 (Leader(t)).
    # If lag=2, shift=1 (Leader(t-1)).
    
    shift_amount = best_lag - 1
    df["a_t_aligned"] = df["a_raw"].shift(shift_amount)
    
    # Leader lagged features (relative to aligned)
    df["a_t_aligned_1"] = df["a_t_aligned"].shift(1)
    df["a_t_aligned_2"] = df["a_t_aligned"].shift(2)
    
    df["a_diff1"] = df["a_t_aligned"] - df["a_t_aligned_1"]
    df["a_pct1"] = (df["a_t_aligned"] - df["a_t_aligned_1"]) / (df["a_t_aligned_1"].replace(0, np.nan) + 1e-6)
    df["a_roll3"] = df["a_t_aligned"].rolling(window=3, min_periods=1).mean()
    
    # Interaction features
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
    
    # Clean up
    # We need to drop rows where features are NaN, but keep rows where target is NaN (for inference)
    # Actually, for training we drop target NaN. For inference we keep it.
    
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
        # We need at least lag+2 history usually
        pair_frame = pair_frame.dropna(subset=["b_t_2", "a_t_aligned_2"])
        frames.append(pair_frame)

    if not frames:
        raise ValueError("No training data created.")
    
    df = pd.concat(frames, ignore_index=True)
    return df

def train_model(train_df: pd.DataFrame):
    # Split
    train_cutoff = pd.Timestamp("2024-12-01")
    valid_start = pd.Timestamp("2025-01-01")
    valid_end = pd.Timestamp("2025-05-01")

    drop_cols = {"date", "target_date", "target_value", "target_log1p", "leading_item_id", "following_item_id", "a_raw"}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    train_mask = (train_df["target_date"] <= train_cutoff) & (train_df["target_log1p"].notna())
    valid_mask = (train_df["target_date"] >= valid_start) & (train_df["target_date"] <= valid_end) & (train_df["target_log1p"].notna())

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "target_log1p"]
    X_valid = train_df.loc[valid_mask, feature_cols]
    y_valid = train_df.loc[valid_mask, "target_log1p"]

    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

    # LGBM with MAE objective (regression_l1)
    model = LGBMRegressor(
        objective="regression_l1", # MAE
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="mae",
        callbacks=[
            # early_stopping is now via callbacks or arguments in recent LGBM
        ]
    )
    
    return model, feature_cols

def build_inference_features(pivot, pairs, forecast_month="2025-08"):
    target_ts = pd.Timestamp(forecast_month + "-01")
    # We need the row where target_date == target_ts
    # In _build_pair_frame, target_date is date + 1 month.
    # So we need the row where date == target_ts - 1 month.
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
        # Select the row for inference
        target_row = pair_frame[pair_frame["date"] == base_ts]
        if not target_row.empty:
            rows.append(target_row.iloc[0])
            
    return pd.DataFrame(rows)

def main():
    print("Loading data...")
    _, pivot = load_monthly_data("train_month.csv")
    
    # 1. Find pairs (Rule-based)
    print("Finding pairs...")
    # Using a slightly lower threshold to get more candidates, or tune this.
    # User used 0.38-0.43. Let's stick to a safe 0.4 for now, or 0.38.
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=15, corr_threshold=0.38)
    print(f"Found {len(pairs)} pairs.")
    
    # 2. Build Data
    print("Building training data...")
    train_df = build_training_data(pivot, pairs)
    
    # 3. Train
    print("Training model...")
    model, feature_cols = train_model(train_df)
    
    # 4. Inference
    print("Inference for 2025-08...")
    pred_df = build_inference_features(pivot, pairs, "2025-08")
    
    if pred_df.empty:
        print("No predictions generated!")
        return

    X_test = pred_df[feature_cols].fillna(0.0)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Inverse log1p
    
    # 5. Submission
    submission = pairs[["leading_item_id", "following_item_id"]].copy()
    submission["value"] = y_pred
    submission["value"] = submission["value"].clip(lower=0).round().astype(int)
    
    # Save
    submission.to_csv("v6_improved_submission.csv", index=False)
    print("Saved v6_improved_submission.csv")

if __name__ == "__main__":
    main()
