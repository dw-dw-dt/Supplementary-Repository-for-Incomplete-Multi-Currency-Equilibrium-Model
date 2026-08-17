import datetime as dt

import numpy as np
import pandas as pd
from numba import njit
from pandas.tseries.offsets import MonthEnd

# パラメータ
# GAMMA_ijはiが下添え字、jが上添え字
GAMMA_11 = 2.0 / 3  # 日本人による日本の財
GAMMA_12 = 1.0 / 3  # 日本人による米国の財
GAMMA_21 = 1.0 / 3  # 米国人による日本の財
GAMMA_22 = 2.0 / 3  # 米国人による米国の財


def get_multivariate_normal_sample(
    RANDOM_SEED: int,
    dim: int,
    dist_mean: np.ndarray,
    dist_cov: np.ndarray,
    size: int = 1,
) -> np.ndarray:
    """Get a sample from a multivariate normal distribution."""
    np.random.seed(RANDOM_SEED)
    dist = np.random.multivariate_normal(dist_mean, dist_cov, size).astype(np.float32)
    if size == 1:
        return dist.reshape(dim, size)  # 3*1
    elif size > 1:
        return dist.reshape(size, dim)  # 100*3
    else:
        raise ValueError("size must be a positive integer.")


@njit
def calculate_alpha(beta: float, t) -> float:
    """Calculate alpha from beta."""
    return np.exp(-beta * t)


@njit
def calculate_y(beta: float, T: int, x_0: float) -> float:
    """Calculate y from beta and x_0."""
    return (1.0 - np.exp(-beta * T)) / (beta * x_0)


@njit
def residual_systematic_resampler(Np: int, weight: np.ndarray, RANDOM_SEED: int):
    """
    Residual systematic resampler.
    :param Np: The number of particles.
    :param weight: The weights of particles.
    """
    np.random.seed(RANDOM_SEED)
    res = [0 for _ in range(Np)]
    u = np.random.rand() / Np
    for p in range(Np):
        res[p] = np.floor((weight[p] - u) * Np) + 1
        u += res[p] / Np - weight[p]
    return res


def load_tp_10_df():
    """
    Load the data of TP1 index.
    """
    tp1_df = pd.read_excel("./data_2/TOPIX.xlsx", sheet_name="Sheet2", header=1).rename(
        columns={"px_last_am": "date"}
    )

    # 日付の列をdatetime型に変換
    tp1_df["date"] = pd.to_datetime(tp1_df["date"])

    # 月末の日付のみ抽出
    tp1_df.set_index("date", inplace=True)
    tp1_df = tp1_df.resample("ME").last()
    tp1_df.index.name = ""
    tp1_df.reset_index(inplace=True, names=["date"])
    return tp1_df


def load_sp_10_df():
    """
    Load the data of SP1 index.
    """
    sp1_df = pd.read_csv("data_2/sp.csv").rename(
        columns={"Dates": "date", "PX_LAST": "value"}
    )
    sp1_df["date"] = pd.to_datetime(sp1_df["date"])
    # 月末の日付のみ抽出
    sp1_df.set_index("date", inplace=True)
    sp1_df = sp1_df.resample("ME").last()
    sp1_df = sp1_df[sp1_df.index >= "2000-01-01"]
    sp1_df.index.name = ""
    sp1_df.reset_index(inplace=True, names=["date"])
    return sp1_df


def load_jp_rate_10_df():
    rate_df = pd.read_csv("data_2/rate.csv")[["date", "MUTSCALM Index"]]
    rate_df.columns = ["date", "value"]
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    # 月末の日付のみ抽出
    rate_df.set_index("date", inplace=True)
    rate_df = rate_df.resample("ME").last()
    rate_df = rate_df[rate_df.index >= "2000-01-01"]
    rate_df.index.name = ""
    rate_df.reset_index(inplace=True, names=["date"])
    return rate_df


def load_us_rate_10_df():
    rate_df = pd.read_csv("data_2/rate.csv")[["date", "FEDL01   Index"]]
    rate_df.columns = ["date", "value"]
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    # 月末の日付のみ抽出
    rate_df.set_index("date", inplace=True)
    rate_df = rate_df.resample("ME").last()
    rate_df = rate_df[rate_df.index >= "2000-01-01"]
    rate_df.index.name = ""
    rate_df.reset_index(inplace=True, names=["date"])
    return rate_df


def load_fx_10_df():
    df = pd.read_csv("data_2/usdjpy_data.csv")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample("ME").last()
    df = df[df.index >= "2000-01-01"]
    df.index.name = ""
    df.reset_index(inplace=True, names=["date"])
    return df


def load_jp_cpi_10_df():
    df = pd.read_excel("data_2/jp_cpi.xlsx", header=7)[
        ["Unnamed: 1", "Unnamed: 12"]
    ].dropna()
    df.columns = ["date", "cpi"]
    df["date"] = pd.to_datetime(
        df["date"].astype(int).astype(str), format="%Y%m"
    ) + MonthEnd(0)
    df["inflation_rate"] = df["cpi"].pct_change()
    df.index = range(len(df))
    return df


def load_us_cpi_10_df():
    df = pd.read_csv("data_2/cpi_data.csv")
    df.columns = ["date", "cpi"]
    df["date"] = pd.to_datetime(df["date"]) + MonthEnd(0)
    df["inflation_rate"] = df["cpi"].pct_change()
    df.loc[0, "inflation_rate"] = 0  # 初期値なのでやむなし
    return df


def load_dfs_11():
    """
    load_dfs_10()の期間を2026年7月まで延長したもの
    """
    jp_cpi_df = load_jp_cpi_10_df()
    us_cpi_df = load_us_cpi_10_df()

    tp1_df = load_tp_10_df()
    tp1_df = pd.merge(tp1_df, jp_cpi_df, on="date", how="left")
    tp1_df["_value"] = tp1_df["value"] / tp1_df["cpi"]
    tp1_df = tp1_df[
        (tp1_df["date"] >= dt.datetime(2000, 1, 1))
        & (tp1_df["date"] <= dt.datetime(2026, 7, 1))
    ]
    tp1_df["scaled_value"] = (
        (tp1_df["_value"] - min(tp1_df["_value"]))
        / (max(tp1_df["_value"]) - min(tp1_df["_value"]))
        * 2
    )

    sp1_df = load_sp_10_df()
    sp1_df = pd.merge(sp1_df, us_cpi_df, on="date", how="left")
    sp1_df["_value"] = sp1_df["value"] / sp1_df["cpi"]
    sp1_df = sp1_df[
        (sp1_df["date"] >= dt.datetime(2000, 1, 1))
        & (sp1_df["date"] <= dt.datetime(2026, 7, 1))
    ]
    sp1_df["scaled_value"] = (
        (sp1_df["_value"] - min(sp1_df["_value"]))
        / (max(sp1_df["_value"]) - min(sp1_df["_value"]))
        * 2
    )

    jprate_df = load_jp_rate_10_df()
    jprate_df = pd.merge(jprate_df, jp_cpi_df, on="date", how="left")
    jprate_df["_value"] = jprate_df["value"] - jprate_df["inflation_rate"]
    jprate_df = jprate_df[
        (jprate_df["date"] >= dt.datetime(2000, 1, 1))
        & (jprate_df["date"] <= dt.datetime(2026, 7, 1))
    ]
    jprate_df["scaled_value"] = (
        (jprate_df["_value"] - min(jprate_df["_value"]))
        / (max(jprate_df["_value"]) - min(jprate_df["_value"]))
        * 2
    )

    usrate_df = load_us_rate_10_df()
    usrate_df = pd.merge(usrate_df, us_cpi_df, on="date", how="left")
    usrate_df["_value"] = usrate_df["value"] - usrate_df["inflation_rate"]
    usrate_df = usrate_df[
        (usrate_df["date"] >= dt.datetime(2000, 1, 1))
        & (usrate_df["date"] <= dt.datetime(2026, 7, 1))
    ]
    usrate_df["scaled_value"] = (
        (usrate_df["_value"] - min(usrate_df["_value"]))
        / (max(usrate_df["_value"]) - min(usrate_df["_value"]))
        * 2
    )

    fx_df = load_fx_10_df()
    fx_df = pd.merge(fx_df, jp_cpi_df, on="date", how="left").rename(
        columns={"cpi": "jp_cpi"}
    )
    fx_df = pd.merge(fx_df, us_cpi_df, on="date", how="left").rename(
        columns={"cpi": "us_cpi"}
    )
    fx_df["_value"] = fx_df["value"] * fx_df["us_cpi"] / fx_df["jp_cpi"]
    fx_df = fx_df[
        (fx_df["date"] >= dt.datetime(2000, 1, 1))
        & (fx_df["date"] <= dt.datetime(2026, 7, 1))
    ]
    fx_df["scaled_value"] = (
        (fx_df["_value"] - min(fx_df["_value"]))
        / (max(fx_df["_value"]) - min(fx_df["_value"]))
        * 2
    )

    date_set = (
        set(tp1_df["date"])
        & set(sp1_df["date"])
        & set(jprate_df["date"])
        & set(usrate_df["date"])
        & set(fx_df["date"])
    )

    tp1_df = tp1_df[tp1_df["date"].isin(date_set)].reset_index(drop=True)
    sp1_df = sp1_df[sp1_df["date"].isin(date_set)].reset_index(drop=True)
    jprate_df = jprate_df[jprate_df["date"].isin(date_set)].reset_index(drop=True)
    usrate_df = usrate_df[usrate_df["date"].isin(date_set)].reset_index(drop=True)
    fx_df = fx_df[fx_df["date"].isin(date_set)].reset_index(drop=True)

    return tp1_df, sp1_df, jprate_df, usrate_df, fx_df


@njit
def project_onto_plane_numba(x, y, z):
    """
    x を y, z の平面に射影する。
    配列はfloat型でなくてはならない。
    intだと np.dot のnumbaがたいおうしていない。
    """
    # ベクトルを列ベクトルとして並べた行列を作成
    A = np.column_stack((y, z))
    # グラム行列を計算 (A.T @ A)
    G = np.dot(A.T, A)
    # A.T @ x を計算
    b = np.dot(A.T, x)
    # 係数 c を計算 (G @ c = b を解く)
    coeffs = np.linalg.solve(G, b)
    # 射影ベクトルを計算
    projection = np.dot(A, coeffs)
    return projection


@njit
def calc_next_delta(delta: np.array, sigma_delta: np.array, dw: np.array, Np: int):
    """
    Calculate the next delta.
    delta: Np * 1
    sigma_delta: dim * 1
    dw: Np * dim
    """
    return (delta + np.dot(dw, sigma_delta) * delta).astype(np.float32).reshape(Np)
