import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_FILE = "data/model3_synthetic_full_50k.csv"

print("\nLoading training dataset...")

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# 2. EXACT FLOOD PREDICTION FEATURES
# =========================================================

FEATURES = [

    # Location
    "latitude",
    "longitude",

    # Terrain
    "elevation_m",
    "hand_m",
    "slope_deg",
    "flow_accumulation",
    "distance_to_river_m",
    "river_network_density_km_km2",

    # Land
    "land_cover",
    "manning_n",

    # Embankments / roads
    "embankment_present",
    "distance_to_embankment_m",
    "embankment_height_m",

    "road_barrier_present",
    "distance_to_major_road_m",

    # Historical/local flood conditions
    "permanent_water_mask",
    "historical_flood_frequency",

    # Soil
    "soil_moisture_pct",

    # Previous rainfall
    "rain_past_6h_mm",
    "rain_past_24h_mm",
    "rain_past_72h_mm",

    # Forecast rainfall
    "forecast_rain_0_6h_mm",
    "forecast_rain_6_12h_mm",
    "forecast_rain_12_24h_mm",
    "forecast_rain_24_48h_mm",

    # River level
    "river_level_now_m",
    "river_level_6h_m",
    "river_level_12h_m",
    "river_level_24h_m",
    "river_level_48h_m",

    # River discharge
    "discharge_now_m3s",
    "discharge_24h_m3s",
    "discharge_48h_m3s",

    # Previous flooding condition
    "previous_flood_depth_m"
]


TARGET = "flooded"


# =========================================================
# 3. CHECK COLUMNS
# =========================================================

missing = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in training dataset: {missing}"
    )


X = df[FEATURES].copy()

y = df[TARGET].astype(int)


print("\nFlood class distribution:")

print(
    y.value_counts()
)

print("\nPercentage:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# =========================================================
# 4. PREPROCESSING
# =========================================================

categorical_features = [
    "land_cover"
]

numeric_features = [
    col for col in FEATURES
    if col not in categorical_features
]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            "passthrough",
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ]

)


# =========================================================
# 5. XGBOOST MODEL
# =========================================================

xgb_model = XGBClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=6,

    min_child_weight=2,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# =========================================================
# 6. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline([

    ("preprocessor", preprocessor),

    ("model", xgb_model)

])


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# =========================================================
# 8. TRAIN
# =========================================================

print("\nTraining XGBoost flood model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================================================
# 9. EVALUATE
# =========================================================

predictions = pipeline.predict(
    X_test
)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n==============================")
print(" MODEL PERFORMANCE")
print("==============================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================================================
# 10. SAVE PIPELINE
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(

    pipeline,

    "models/flood_pipeline.pkl"

)


print("\n==============================")
print(" TRAINING COMPLETED")
print("==============================")

print(
    "Saved: models/flood_pipeline.pkl"
)