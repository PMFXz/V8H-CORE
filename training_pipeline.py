import joblib
from .config_training import PATHS, TIMEFRAMES
from .config_training import config
from .dataset_loader import DatasetLoader
from .feature_set_core import FeatureSetCore
from .model_definition import build_model
from .trainer_v8h import TrainerV8H
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def run_pipeline():
    # 1) โหลดดาต้า
    loader = DatasetLoader()
    for tf in TIMEFRAMES:
        print(f"Processing timeframe:{tf}")
        # load raw
        df_raw = loader.load_raw(tf)

        # clean
        df_clean = loader.clean_data(df_raw)

        # save clean
        loader.save_clean(df_clean, tf)
    print("Pipeline completed successfully.")
    # 2) ทำฟีเจอร์
    fe = FeatureSetCore()
    df = loader.load_raw("M15")
    FEATURE_COLS = config["training"]["feature_columns"]

    df_features = fe.build_features(df_clean)
    # new label
    df_features = fe.create_label(df_features)
    # drop noise
    df_features = df_features.dropna(subset=["y"])

    df_features = df_features[FEATURE_COLS]
    # create new label trend risk zone
    df_features = fe.create_multi_label(df_features)
    X = df_features[FEATURE_COLS]
    y_trend = df_features["y_trend"]
    y_zone = df_features["y_zone"]
    y_risk = df_features["y_risk"]

    # Fix Input dim
    input_dim = len(FEATURE_COLS)

    # 3) สร้างโมเดล
    model = build_model(input_dim)

    # 4) เทรนโมเดล

    # Core V8H
    trainer = TrainerV8H(model, config)
    trainer.train(df_features)

    # Scaler Model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[OK] scaler.pkl saved")

    # Trend Model
    trend_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    trend_model.fit(X_scaled, y_trend)
    joblib.dump(trend_model, "models/trend_model.pkl")
    print("[OK] trend_model.pkl saved")

    # Zone Model
    zone_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    zone_model.fit(X_scaled, y_zone)
    joblib.dump(zone_model, "models/zone_model.pkl")
    print("[OK] zone_model.pkl saved")

    # Risk Model
    risk_model = LogisticRegression(max_iter=1000)
    risk_model.fit(X_scaled, y_risk)
    joblib.dump(risk_model, "models/risk_model.pkl")
    print("[OK] risk_model.pkl saved")

    # 5) เซฟโมเดล
    trainer.save_model(config["paths"]["models"])

    print("TRAINING COMPLETE.")


if __name__ == "__main__":
    run_pipeline()
