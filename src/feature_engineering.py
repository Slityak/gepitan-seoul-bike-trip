"""Feature engineering: ex-ante (induláskor ismert) feature-ök építése.

Stateful FeatureEngineer (KMeans zónák tanulása) + ColumnTransformer
(skálázás, OneHot/TargetEncoding). Kimenet: numpy mátrix + feature-nevek.

Dobott oszlopok (post-trip / id-szerű leakage):
    - "Distance"      — GPS-trace tényleges hossza, csak a trip végén ismert.
    - "Unnamed: 0"    — CSV row index, kvázi-leakage.
    - "GroundTemp"    — a forrásban hiányos / redundáns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Seoul középszélessége — fokokat ezzel skálázzuk hozzávetőleg méterre.
SEOUL_LAT_DEG = 37.55
DEG_TO_M_LAT = 111_000.0
DEG_TO_M_LONG = float(np.cos(np.deg2rad(SEOUL_LAT_DEG)) * DEG_TO_M_LAT)

RUSH_HOURS = frozenset({7, 8, 9, 17, 18, 19})
LEAKAGE_OR_NOISE_COLS = ("Unnamed: 0", "GroundTemp", "Distance")


NUMERIC_FEATURES: list[str] = [
    # térbeli (Distance és route_directness post-trip → kihagyva)
    "Haversine_log",
    "manhattan_dist_m",
    "dlat_m",
    "dlon_m",
    "bearing_sin",
    "bearing_cos",
    "PLatd",
    "PLong",
    "DLatd",
    "DLong",
    # idő — ordinal és cyclic párhuzamosan (a fák az ordinalt, a lineárisak a ciklikust szeretik)
    "Pmonth",
    "Pday",
    "Phour",
    "Pmin",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos",
    # időjárás
    "Temp",
    "Wind",
    "Humid",
    "Solar",
    "Precip_log",
    "Snow_log",
    "Dust_log",
    # interakciók (Haversine az ex-ante távolságproxy)
    "haversine_x_rush",
    "haversine_x_weekend",
]

BOOL_FEATURES: list[str] = [
    "is_rush_hour",
    "is_weekend",
    "is_rainy",
    "is_snowy",
    "bad_weather",
]

CATEGORICAL_FEATURES: list[str] = ["PDweek", "start_zone", "end_zone"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateful feature-építő: train-en KMeans zónákat tanul.

    Nyers DataFrame-et vár (load_train_test X-kimenete). DataFrame-et ad vissza,
    a downstream ColumnTransformer ezen választja ki az oszlopokat névszerint.
    """

    def __init__(self, n_zones: int = 20, random_state: int = 42) -> None:
        self.n_zones = n_zones
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        self.start_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=self.random_state,
            batch_size=4096,
            n_init=3,
        ).fit(X[["PLatd", "PLong"]].to_numpy())
        self.end_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=self.random_state,
            batch_size=4096,
            n_init=3,
        ).fit(X[["DLatd", "DLong"]].to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in LEAKAGE_OR_NOISE_COLS:
            if c in df.columns:
                df = df.drop(columns=c)

        # Térbeli: méter-skálájú deltak, manhattan, bearing
        dlat = (df["DLatd"] - df["PLatd"]).to_numpy()
        dlon = (df["DLong"] - df["PLong"]).to_numpy()
        dlat_m = dlat * DEG_TO_M_LAT
        dlon_m = dlon * DEG_TO_M_LONG
        df["dlat_m"] = dlat_m
        df["dlon_m"] = dlon_m
        df["manhattan_dist_m"] = np.abs(dlat_m) + np.abs(dlon_m)

        bearing = np.arctan2(dlon_m, dlat_m)  # 0 = észak, π/2 = kelet
        df["bearing_sin"] = np.sin(bearing)
        df["bearing_cos"] = np.cos(bearing)

        # Haversine km → m, log
        haversine_m = (df["Haversine"].to_numpy() * 1000.0).clip(min=0.0)
        df["Haversine_log"] = np.log1p(haversine_m)

        # Időjárás log1p (heavy-tailed, sok 0)
        for col in ("Precip", "Snow", "Dust"):
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0).to_numpy())

        df["is_rainy"] = (df["Precip"] > 0).astype(np.int8)
        df["is_snowy"] = (df["Snow"] > 0).astype(np.int8)
        df["bad_weather"] = ((df["Precip"] > 0) | (df["Snow"] > 0)).astype(np.int8)

        # Idő: cyclic encoding mindenhova, ahol értelmes
        h = df["Phour"].to_numpy()
        df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)

        m = df["Pmonth"].to_numpy()
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

        d = df["PDweek"].to_numpy()
        df["dow_sin"] = np.sin(2 * np.pi * d / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * d / 7.0)

        df["is_rush_hour"] = df["Phour"].isin(RUSH_HOURS).astype(np.int8)
        df["is_weekend"] = (df["PDweek"] >= 5).astype(np.int8)

        # Interakciók: a Haversine az ex-ante távolság-proxy
        df["haversine_x_rush"] = df["Haversine_log"] * df["is_rush_hour"]
        df["haversine_x_weekend"] = df["Haversine_log"] * df["is_weekend"]

        # KMeans zónák — train-en fit-elve
        df["start_zone"] = self.start_kmeans_.predict(
            df[["PLatd", "PLong"]].to_numpy()
        ).astype(np.int16)
        df["end_zone"] = self.end_kmeans_.predict(
            df[["DLatd", "DLong"]].to_numpy()
        ).astype(np.int16)

        df["PDweek"] = df["PDweek"].astype(np.int16)
        return df


def build_preprocessor(use_target_encoding: bool = False) -> ColumnTransformer:
    """ColumnTransformer: numerikusra RobustScaler, kategórikusra OneHot vagy TargetEncoder."""
    transformers: list = [
        ("num", RobustScaler(), NUMERIC_FEATURES),
        ("bool", "passthrough", BOOL_FEATURES),
    ]

    if use_target_encoding:
        from sklearn.preprocessing import TargetEncoder

        transformers.append((
            "cat",
            TargetEncoder(
                target_type="continuous",
                smooth="auto",
                cv=5,
                random_state=42,
            ),
            CATEGORICAL_FEATURES,
        ))
    else:
        transformers.append((
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            CATEGORICAL_FEATURES,
        ))

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    use_target_encoding: bool = False,
    n_zones: int = 20,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("fe", FeatureEngineer(n_zones=n_zones, random_state=random_state)),
        ("pre", build_preprocessor(use_target_encoding=use_target_encoding)),
    ])
