# =========================================================
#  FEATURE SET CORE (Unified for Training & Inference)
#  V8H-CORE | PMFX TRADING COMPANY
# =========================================================

import pandas as pd
import numpy as np


class FeatureSetCore:
    def __init__(self, config=None):
        self.config = config
        pass

    # -------------------------------
    # Basic indicators
    # -------------------------------
    def ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def lwma(self, series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def atr(self, df, period=14):
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # -------------------------------
    # Trend Features
    # -------------------------------
    def add_trend_features(self, df):
        df = df.copy()
        # EMA
        df["ema_20"] = self.ema(df["close"], 20)
        df["ema_50"] = self.ema(df["close"], 50)
        df["ema_200"] = self.ema(df["close"], 200)
        # LWMA
        df["lwma_7"] = self.lwma(df["close"], 7)
        df["lwma_20"] = self.lwma(df["close"], 20)
        df["lwma_60"] = self.lwma(df["close"], 60)
        # ATR
        df["atr_14"] = self.atr(df, 14)
        # Trend slope
        df["trend_slope"] = df["close"].diff(5)
        # Above EMA200
        df["above_ema200"] = (df["close"] > df["ema_200"]).astype(int)

        return df

    # -------------------------------
    # Zone Features (BIG PRICE ZONES)
    # -------------------------------
    def add_zone_features(self, df, zone_step=50):
        df = df.copy()
        df["zone"] = (df["close"] // zone_step) * zone_step
        df["distance_from_zone"] = df["close"] - df["zone"]
        df["near_zone"] = (df["distance_from_zone"].abs() < 10).astype(int)

        return df

    # -------------------------------
    # Volatility & Risk Features
    # -------------------------------
    def add_vol_risk_features(self, df):
        df = df.copy()
        # Rolling volatility (20 periods)
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        # Volume MA
        if "volume" in df.columns:
            df["volume_ma"] = self.ema(df["volume"], 20)
        else:
            df["volume_ma"] = 0  # MT5 ticks ไม่มี volume ให้ default
        return df

    # -------------------------------
    # Create Label for ML
    # -------------------------------
    def create_label(self, df, lookahead=5):
        df = df.copy()
        future_close = df["close"].shift(-lookahead)
        threshold = df["atr_14"] * 0.3
        df["y"] = (future_close - df["close"] > threshold).astype(int)
        df.loc[(df["close"] - future_close > threshold), "y"] = 0
        noise_mask = (future_close - df["close"]).abs() <= threshold
        df.loc[noise_mask, "y"] = None

        return df

    def create_multi_label(self, df, lookahead=5):
        df = df.copy()
        # y_trend
        df["y_trend"] = (df["close"].shift(-lookahead) > df["close"]).astype(int)
        # y_zone
        future_distance = (df["close"].shift(-lookahead) - df["zone"]).abs()
        df["y_zone"] = (future_distance < 10).astype(int)
        # y_risk
        df["y_risk"] = (df["atr_14"].shift(-lookahead) > df["atr_14"]).astype(int)
        return df.dropna()

    # -------------------------------
    # Final Combined Feature Set
    # -------------------------------
    def build_features(self, df):
        df = df.copy()
        df = self.add_trend_features(df)
        df = self.add_zone_features(df)
        df = self.add_vol_risk_features(df)
        # Remove NaN indicators
        df = df.dropna()
        return df


class FeatureSetCoreRealtime:
    def __init__(self, window=200):
        self.window = window
        self.buffer = []

    def update(self, tick: dict):
        # tick = {"bid":...,"ask":...,"volume":...}
        price = (tick["bid"] + tick["ask"]) / 2
        vol = tick.get("volume", 0)
        self.buffer.append(
            {
                "open": tick.get("open", "price"),
                "high": tick.get("high", "price"),
                "low": tick.get("low", "price"),
                "close": price,
                "volume": vol,
            }
        )
        # collect last window
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def to_numpy(self):
        if len(self.buffer) < 200:
            return None  # wait for buffer 50 candle
        df = pd.DataFrame(self.buffer)
        # use feature
        # EMA
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()

        # LWMA
        def lwma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )

        df["lwma_7"] = lwma(df["close"], 7)
        df["lwma_20"] = lwma(df["close"], 20)
        df["lwma_60"] = lwma(df["close"], 60)
        # ATR
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["close_shift"] = df["close"].shift()
        tr = pd.concat(
            [
                (df["high"] - df["low"]),
                (df["high"] - df["close_shift"]).abs(),
                (df["low"] - df["close_shift"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        # Trend slope
        df["trend_slope"] = df["close"].diff(5)
        # Above EMA200
        df["above_ema200"] = (df["close"] > df["ema_200"]).astype(int)
        # ==========
        # Zone Features
        # ==========
        df["zone"] = (df["close"] // 50) * 50
        df["distance_from_zone"] = df["close"] - df["zone"]
        df["near_zone"] = (df["distance_from_zone"].abs() < 10).astype(int)
        # ==========
        # Volatility & Risk
        # ==========
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        df["volume_ma"] = df["volume"].ewm(span=20).mean()
        # ==========
        # CLEAN
        # ==========
        df = df.dropna()
        # Deugging Logger :
        print("Length of Dataframe after dropna :", len(df))
        print("Data after dropna tail:", df.tail())

        # ถ้าฟีเจอร์ไม่ครบ ยังไม่พร้อม
        if len(df) == 0:
            return None
        # ==========
        # Feature columns ตาม TRAINING
        # ==========
        feature_cols = [
            # core OHLCV
            "open",
            "high",
            "low",
            "close",
            "volume",
            # --- Trend / MA Features ---
            "ema_20",
            "ema_50",
            "ema_200",
            "lwma_7",
            "lwma_20",
            "lwma_60",
            # --- Votility ---
            "atr_14",
            # --- Trend slope ---
            "trend_slope",
            # --- Above EMA 200 (binary) ---
            "above_ema200",
            # --- Zone Features ---
            "zone",
            "distance_from_zone",
            "near_zone",
        ]
        row = df.iloc[-1:][feature_cols].values.astype(np.float32)
        # reshape → (1, features)
        return row.astype(np.float32).reshape(1, -1)
