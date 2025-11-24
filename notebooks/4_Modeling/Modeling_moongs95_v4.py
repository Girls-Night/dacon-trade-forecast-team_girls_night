"""공행 규칙 + LightGBM OOF 앙상블 예측 파이프라인(v4).

1. train_month.csv을 불러 item_id × ym pivot 생성
2. 상관계수 규칙으로 (leader, follower) 공행쌍 탐색
3. follower/leader lag, diff, ratio 파생 피처 생성 후 시간순 OOF(TimeSeriesSplit) LightGBM 앙상블 학습
4. 2025-08 월의 pair value 예측
5. submission_{tag}_{오늘날짜}.csv 저장
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. 월별 데이터 읽기 및 pivot 생성
# ---------------------------------------------------------------------------

def load_monthly_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train_month.csv을 읽어 item_id × ym pivot 생성."""

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
# 1. 공행쌍 탐색 (상관 기반)
# ---------------------------------------------------------------------------

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """분산=0 예외처리 포함한 상관계수."""

    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def find_comovement_pairs(
    pivot: pd.DataFrame,
    max_lag: int = 6,
    min_nonzero: int = 12,
    corr_threshold: float = 0.4,
) -> pd.DataFrame:
    """아이템별 최대 상관 lag를 찾아 (leader, follower) 공행쌍 추출."""

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
# 2. 학습/추론용 피처 구성
# ---------------------------------------------------------------------------

def _build_pair_frame(
    pivot: pd.DataFrame,
    leader: str,
    follower: str,
    best_lag: int,
    max_corr: float,
    for_inference: bool = False,
) -> pd.DataFrame:
    """단일 pair에 대해 시계열 파생 피처 계산 (학습/추론 겸용)."""

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
    ]
    if for_inference:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 추론에서는 마지막 시점 유지를 위해 결측을 드롭하지 않고 이후 단계에서 fillna 처리
        return df

    df = df.dropna(subset=required_cols + ["target_value"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def build_training_data(pivot: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """모든 pair의 학습용 피처 프레임 결합."""

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
        raise ValueError("학습 피처가 비어 있습니다.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["target_date"].notna()]
    return df


# ---------------------------------------------------------------------------
# 3. LightGBM OOF TimeSeriesSplit 앙상블 학습
# ---------------------------------------------------------------------------

def train_oof_models(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[LGBMRegressor], List[str], List[float], float]:
    """시간 순서 기반 TimeSeriesSplit OOF로 모델을 학습하고 fold별/전체 성능을 반환."""

    drop_cols = {"date", "target_date", "target_value", "target_log1p", "leading_item_id", "following_item_id"}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    # 시간 누수를 막기 위해 target_date 기준 정렬 후 순차 분할
    train_df_sorted = train_df.sort_values("target_date").reset_index(drop=True)

    y_all = train_df_sorted["target_log1p"].values
    oof_pred = np.zeros(len(train_df_sorted))
    models: List[LGBMRegressor] = []
    fold_scores: List[float] = []

    tscv = TimeSeriesSplit(n_splits=n_splits)
    pbar = tqdm(tscv.split(train_df_sorted), total=n_splits, desc="TimeSeriesSplit 학습", unit="fold")

    for fold, (tr_idx, val_idx) in enumerate(pbar, 1):
        start = time.time()
        X_train = train_df_sorted.iloc[tr_idx][feature_cols]
        y_train = train_df_sorted.iloc[tr_idx]["target_log1p"]
        X_valid = train_df_sorted.iloc[val_idx][feature_cols]
        y_valid = train_df_sorted.iloc[val_idx]["target_log1p"]

        model = LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[early_stopping(100, verbose=False)],
        )

        best_iter = model.best_iteration_ or model.n_estimators
        pred_val = model.predict(X_valid, num_iteration=best_iter)
        oof_pred[val_idx] = pred_val

        rmse = float(np.sqrt(np.mean((pred_val - y_valid) ** 2)))
        fold_scores.append(rmse)
        models.append(model)

        elapsed = time.time() - start
        pbar.set_postfix(best_iter=best_iter, rmse=f"{rmse:.4f}", secs=f"{elapsed:.1f}")

    overall_rmse = float(np.sqrt(np.mean((oof_pred - y_all) ** 2)))
    tqdm.write(f"OOF 전체 RMSE: {overall_rmse:.4f}")
    return models, feature_cols, fold_scores, overall_rmse


# ---------------------------------------------------------------------------
# 4. 추론용 피처 생성
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
            for_inference=True,
        )
        target_row = pair_frame[pair_frame["target_date"] == target_ts]
        if not target_row.empty:
            rows.append(target_row.iloc[0].copy())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. 앙상블 예측 및 제출 파일 생성
# ---------------------------------------------------------------------------

def _predict_ensemble(models: List[LGBMRegressor], X: pd.DataFrame) -> np.ndarray:
    """개별 모델 예측을 평균하여 앙상블 출력."""

    preds = []
    for model in models:
        num_iter = getattr(model, "best_iteration_", None)
        preds.append(model.predict(X, num_iteration=num_iter))
    return np.mean(preds, axis=0)


def create_submission(
    pairs: pd.DataFrame,
    pred_df: pd.DataFrame,
    pivot: pd.DataFrame,
    models: List[LGBMRegressor],
    feature_cols: List[str],
) -> pd.DataFrame:
    """pair 목록과 예측 value를 결합해 submission 생성."""

    submission = pairs[["leading_item_id", "following_item_id"]].drop_duplicates().reset_index(drop=True)
    latest_map = pivot.iloc[-1].to_dict()

    if pred_df.empty:
        submission["value"] = submission["following_item_id"].map(latest_map).fillna(0.0)
    else:
        X_test = pred_df[feature_cols].fillna(0.0)
        y_pred = _predict_ensemble(models, X_test)
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
    forecast_month = "2025-08"  # 마지막 관측(2025-07) 이후 값 예측
    feature_tag = "rulelgbm_v4_oof"

    start_time = time.time()
    _, pivot = load_monthly_data(data_path)
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4)
    if pairs.empty:
        raise ValueError("공행쌍을 찾지 못했습니다.")

    train_df = build_training_data(pivot, pairs)
    models, feature_cols, fold_scores, oof_rmse = train_oof_models(train_df, n_splits=5, random_state=42)

    pred_df = build_inference_features(pivot, pairs, forecast_month)
    submission = create_submission(pairs, pred_df, pivot, models, feature_cols)

    today = pd.Timestamp.today().strftime("%Y%m%d")
    output_path = f"submission_{feature_tag}_{today}.csv"
    submission.to_csv(output_path, index=False)

    total_mins = (time.time() - start_time) / 60
    print(f"{output_path} 저장 완료 (총 {len(submission)}개 pair, OOF RMSE={oof_rmse:.4f}, 약 {total_mins:.1f}분 소요)")


if __name__ == "__main__":
    main()
