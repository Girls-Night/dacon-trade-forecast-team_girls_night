"""룰 기반 공행성 탐색 + LightGBM 회귀 예측 파이프라인 (v3).

1. train_month.csv를 월별 value 합계 pivot으로 변환
2. 피어슨 상관계수 규칙으로 (leader, follower) 공행성 쌍 탐색
3. follower/leader lag, diff, ratio 등 피처 생성 후 LightGBM 회귀 학습
4. 2025-08을 대상으로 pair별 value 예측
5. submission_{tag}_{오늘날짜}.csv 저장
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. 데이터 적재 및 월별 pivot 생성
# ---------------------------------------------------------------------------

def load_monthly_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# 1. 공행성 쌍 탐색 (룰 기반)
# ---------------------------------------------------------------------------

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """분산=0 예외를 처리한 피어슨 상관계수."""

    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def find_comovement_pairs(
    pivot: pd.DataFrame,
    max_lag: int = 6,
    min_nonzero: int = 12,
    corr_threshold: float = 0.4,
) -> pd.DataFrame:
    """피어슨 룰 기반으로 (leader, follower) 공행성 쌍 탐색."""

    items = pivot.columns.tolist()
    pairs: List[Dict[str, float]] = []

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
            for lag in range(1, max_lag + 1):
                if len(leader_series) <= lag:
                    break
                x = leader_series[:-lag]
                y = follower_series[lag:]
                corr = safe_corr(x, y)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            if best_lag is None or abs(best_corr) < corr_threshold:
                continue

            pairs.append(
                {
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": int(best_lag),
                    "max_corr": float(best_corr),
                }
            )

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# 2. 회귀용 학습 데이터 구성
# ---------------------------------------------------------------------------

def _build_pair_frame(
    pivot: pd.DataFrame,
    leader: str,
    follower: str,
    best_lag: int,
    max_corr: float,
) -> pd.DataFrame:
    """단일 pair에 대한 시점별 피처 계산."""

    leader_series = pivot[leader]
    follower_series = pivot[follower]

    df = pd.DataFrame(
        {
            "date": pivot.index,
            "b_t": follower_series.values,
        }
    )
    df["b_t_1"] = df["b_t"].shift(1)
    df["b_t_2"] = df["b_t"].shift(2)
    df["b_roll3"] = follower_series.rolling(window=3, min_periods=3).mean().values
    df["b_diff1"] = df["b_t"] - df["b_t_1"]
    df["b_pct1"] = (df["b_t"] - df["b_t_1"]) / (df["b_t_1"].replace(0, np.nan) + 1e-6)

    df["a_t_lag"] = leader_series.shift(best_lag).values
    df["a_t_lag_1"] = leader_series.shift(best_lag + 1).values
    df["a_diff_lag"] = df["a_t_lag"] - df["a_t_lag_1"]
    df["a_pct_lag"] = (df["a_t_lag"] - df["a_t_lag_1"]) / (df["a_t_lag_1"].replace(0, np.nan) + 1e-6)

    df["ratio_ab_t"] = df["a_t_lag"] / (df["b_t"] + 1.0)
    df["spread_ab_t"] = df["a_t_lag"] - df["b_t"]

    df["target_value"] = df["b_t"].shift(-1)
    df["target_log1p"] = np.log1p(df["target_value"].clip(lower=0))
    df["target_date"] = df["date"] + pd.offsets.MonthBegin(1)

    df["leading_item_id"] = leader
    df["following_item_id"] = follower
    df["best_lag"] = best_lag
    df["max_corr"] = max_corr

    required_cols = [
        "b_t",
        "b_t_1",
        "b_t_2",
        "b_roll3",
        "b_diff1",
        "b_pct1",
        "a_t_lag",
        "a_t_lag_1",
        "a_diff_lag",
        "a_pct_lag",
        "target_value",
    ]
    df = df.dropna(subset=required_cols)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def build_training_data(pivot: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """모든 pair에 대한 학습용 레코드 결합."""

    frames = []
    for _, row in pairs.iterrows():
        pair_frame = _build_pair_frame(
            pivot,
            leader=row["leading_item_id"],
            follower=row["following_item_id"],
            best_lag=row["best_lag"],
            max_corr=row["max_corr"],
        )
        if not pair_frame.empty:
            frames.append(pair_frame)

    if not frames:
        raise ValueError("학습 데이터가 생성되지 않았습니다.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["target_date"].notna()]
    return df


# ---------------------------------------------------------------------------
# 3. LightGBM 회귀 모델 학습
# ---------------------------------------------------------------------------

def train_regression_model(train_df: pd.DataFrame) -> Tuple[LGBMRegressor, List[str]]:
    """시간 기준 train/valid 분할 후 LightGBM 학습."""

    train_cutoff = pd.Timestamp("2024-12-01")
    valid_start = pd.Timestamp("2025-01-01")
    valid_end = pd.Timestamp("2025-05-01")

    drop_cols = {"date", "target_date", "target_value", "target_log1p", "leading_item_id", "following_item_id"}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    train_mask = train_df["target_date"] <= train_cutoff
    valid_mask = (train_df["target_date"] >= valid_start) & (train_df["target_date"] <= valid_end)

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("train/valid 기간 데이터가 부족합니다.")

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "target_log1p"]
    X_valid = train_df.loc[valid_mask, feature_cols]
    y_valid = train_df.loc[valid_mask, "target_log1p"]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="l2")
    return model, feature_cols


# ---------------------------------------------------------------------------
# 4. 예측 월 피처 생성
# ---------------------------------------------------------------------------

def build_inference_features(
    pivot: pd.DataFrame,
    pairs: pd.DataFrame,
    forecast_month: str,
) -> pd.DataFrame:
    """forecast_month(예: 2025-08)에 대한 pair별 입력 피처 생성."""

    target_ts = pd.Timestamp(forecast_month + "-01")
    base_ts = target_ts - pd.offsets.MonthBegin(1)
    if base_ts not in pivot.index:
        raise ValueError(f"{base_ts.date()} 기준 데이터가 없습니다.")

    rows = []
    for _, row in pairs.iterrows():
        pair_frame = _build_pair_frame(
            pivot,
            leader=row["leading_item_id"],
            follower=row["following_item_id"],
            best_lag=row["best_lag"],
            max_corr=row["max_corr"],
        )
        target_row = pair_frame[pair_frame["target_date"] == target_ts]
        if not target_row.empty:
            rows.append(target_row.iloc[0].copy())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. 제출 파일 생성
# ---------------------------------------------------------------------------

def create_submission(
    pairs: pd.DataFrame,
    pred_df: pd.DataFrame,
    pivot: pd.DataFrame,
    model: LGBMRegressor,
    feature_cols: List[str],
) -> pd.DataFrame:
    """pair 목록에 예측 value를 결합하여 submission 생성."""

    submission = pairs[["leading_item_id", "following_item_id"]].drop_duplicates().reset_index(drop=True)
    latest_map = pivot.iloc[-1].to_dict()

    if pred_df.empty:
        submission["value"] = submission["following_item_id"].map(latest_map).fillna(0.0)
    else:
        X_test = pred_df[feature_cols].fillna(0.0)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(0.0, np.expm1(y_pred))
        pred_values = pred_df[["leading_item_id", "following_item_id"]].copy()
        pred_values["value"] = y_pred

        submission = submission.merge(
            pred_values,
            on=["leading_item_id", "following_item_id"],
            how="left",
        )
        submission["value"] = submission["value"].fillna(
            submission["following_item_id"].map(latest_map)
        )

    submission["value"] = submission["value"].fillna(0.0)
    submission["value"] = submission["value"].clip(lower=0).round().astype(int)
    submission.drop_duplicates(["leading_item_id", "following_item_id"], inplace=True)
    submission.reset_index(drop=True, inplace=True)
    return submission


# ---------------------------------------------------------------------------
# 6. 메인 실행부
# ---------------------------------------------------------------------------

def main():
    data_path = "train_month.csv"
    forecast_month = "2025-08"  # 마지막 관측(2025-07) 이후 한 달 예측
    feature_tag = "rulelgbm_v3"

    _, pivot = load_monthly_data(data_path)
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4)
    if pairs.empty:
        raise ValueError("공행성 쌍이 탐색되지 않았습니다.")

    train_df = build_training_data(pivot, pairs)
    model, feature_cols = train_regression_model(train_df)

    pred_df = build_inference_features(pivot, pairs, forecast_month)
    submission = create_submission(pairs, pred_df, pivot, model, feature_cols)

    today = pd.Timestamp.today().strftime("%Y%m%d")
    output_path = f"submission_{feature_tag}_{today}.csv"
    submission.to_csv(output_path, index=False)
    print(f"{output_path} 저장 완료 (총 {len(submission)}개 pair)")


if __name__ == "__main__":
    main()
