import joblib
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from impact_analysis import analyze_impact


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "models/flood_pipeline.pkl"

SYNTHETIC_FILE = (
    "data/model3_synthetic_full_50k.csv"
)

VILLAGE_FILE = (
    "data/bihar_9district_villages.geojson"
)

OUTPUT_FILE = (
    "village_flood_predictions.csv"
)


# =========================================================
# SAME FEATURES USED WHILE TRAINING
# =========================================================

FEATURES = [

    "latitude",
    "longitude",

    "elevation_m",
    "hand_m",
    "slope_deg",
    "flow_accumulation",
    "distance_to_river_m",
    "river_network_density_km_km2",

    "land_cover",
    "manning_n",

    "embankment_present",
    "distance_to_embankment_m",
    "embankment_height_m",

    "road_barrier_present",
    "distance_to_major_road_m",

    "permanent_water_mask",
    "historical_flood_frequency",

    "soil_moisture_pct",

    "rain_past_6h_mm",
    "rain_past_24h_mm",
    "rain_past_72h_mm",

    "forecast_rain_0_6h_mm",
    "forecast_rain_6_12h_mm",
    "forecast_rain_12_24h_mm",
    "forecast_rain_24_48h_mm",

    "river_level_now_m",
    "river_level_6h_m",
    "river_level_12h_m",
    "river_level_24h_m",
    "river_level_48h_m",

    "discharge_now_m3s",
    "discharge_24h_m3s",
    "discharge_48h_m3s",

    "previous_flood_depth_m"
]


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained XGBoost model...")

model = joblib.load(
    MODEL_FILE
)


# =========================================================
# LOAD SYNTHETIC ENVIRONMENT DATA
# =========================================================

print("Loading environmental prototype data...")

synthetic = pd.read_csv(
    SYNTHETIC_FILE
)


missing = [
    col for col in FEATURES
    if col not in synthetic.columns
]

if missing:
    raise ValueError(
        f"Missing environmental columns: {missing}"
    )


# =========================================================
# LOAD REAL VILLAGE BOUNDARIES
# =========================================================

print("Loading village boundaries...")

villages = gpd.read_file(
    VILLAGE_FILE
)

villages = villages.to_crs(
    "EPSG:4326"
)

print(
    "Total villages:",
    len(villages)
)


# =========================================================
# MATCH EACH VILLAGE TO NEAREST ENVIRONMENTAL RECORD
# =========================================================

print("\nMatching environmental conditions...")

environment_coordinates = synthetic[
    ["latitude", "longitude"]
].to_numpy()

village_coordinates = villages[
    ["latitude", "longitude"]
].to_numpy()


nearest_model = NearestNeighbors(
    n_neighbors=1,
    algorithm="ball_tree"
)

nearest_model.fit(
    environment_coordinates
)


distances, indices = nearest_model.kneighbors(
    village_coordinates
)

nearest_indices = indices.flatten()


prediction_data = synthetic.iloc[
    nearest_indices
][FEATURES].copy()

prediction_data.reset_index(
    drop=True,
    inplace=True
)


# Use actual real village coordinates
prediction_data["latitude"] = (
    villages["latitude"].to_numpy()
)

prediction_data["longitude"] = (
    villages["longitude"].to_numpy()
)


# =========================================================
# FLOOD PREDICTION
# =========================================================

print(
    "Running flood prediction for all villages..."
)

probabilities = (
    model.predict_proba(
        prediction_data
    )[:, 1]
    * 100
)


# =========================================================
# FLOOD RISK LEVEL
# =========================================================

risk_levels = np.where(

    probabilities >= 75,
    "HIGH",

    np.where(
        probabilities >= 40,
        "MODERATE",
        "LOW"
    )
)


# =========================================================
# CREATE RESULTS TABLE
# =========================================================

results = pd.DataFrame({

    "village_name":
        villages["village_name"].values,

    "block_name":
        villages["block_name"].values,

    "district":
        villages["district"].values,

    "latitude":
        villages["latitude"].values,

    "longitude":
        villages["longitude"].values,

    "flood_probability":
        np.round(probabilities, 2),

    "risk_level":
        risk_levels,

    "population_affected": 0,

    "schools_affected": 0,

    "hospitals_affected": 0,

    "markets_affected": 0,

    "total_places_affected": 0
})


# =========================================================
# PROBABILITY STATISTICS
# =========================================================

print("\n==============================")
print(" FLOOD PROBABILITY STATISTICS")
print("==============================")

print(
    results["flood_probability"].describe()
)


# =========================================================
# GIS IMPACT ANALYSIS
# Only MODERATE/HIGH villages
# =========================================================

risk_indices = results.index[
    results["flood_probability"] >= 40
].tolist()


print(
    "\nVillages requiring impact analysis:",
    len(risk_indices)
)


for count, i in enumerate(
    risk_indices,
    start=1
):

    geometry = villages.iloc[i].geometry

    impact = analyze_impact(
        geometry
    )

    results.at[
        i,
        "population_affected"
    ] = impact[
        "population_affected"
    ]

    results.at[
        i,
        "schools_affected"
    ] = impact[
        "schools_affected"
    ]

    results.at[
        i,
        "hospitals_affected"
    ] = impact[
        "hospitals_affected"
    ]

    results.at[
        i,
        "markets_affected"
    ] = impact[
        "markets_affected"
    ]

    results.at[
        i,
        "total_places_affected"
    ] = impact[
        "total_places_affected"
    ]


    if count % 100 == 0:

        print(
            f"Impact analysis: "
            f"{count}/{len(risk_indices)}"
        )


# =========================================================
# NORMALIZATION FUNCTION
# Converts impact values to 0-100
# =========================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
        * 100
    )


# =========================================================
# NORMALIZED IMPACT VALUES
# =========================================================

population_score = normalize(
    results["population_affected"]
)

hospital_score = normalize(
    results["hospitals_affected"]
)

school_score = normalize(
    results["schools_affected"]
)

market_score = normalize(
    results["markets_affected"]
)


# =========================================================
# PRIORITY SCORE
#
# 45% Flood probability
# 30% Population impact
# 10% Hospital impact
# 10% School impact
# 5% Market impact
# =========================================================

results["priority_score"] = (

    0.45
    * results["flood_probability"]

    +

    0.30
    * population_score

    +

    0.10
    * hospital_score

    +

    0.10
    * school_score

    +

    0.05
    * market_score

).round(2)


# LOW flood probability should never receive
# operational priority because of population alone.

results.loc[
    results["flood_probability"] < 40,
    "priority_score"
] = 0


# =========================================================
# ACTION PRIORITY
# =========================================================

def get_action_priority(row):

    probability = row[
        "flood_probability"
    ]

    score = row[
        "priority_score"
    ]


    # No significant predicted flood
    if probability < 40:

        return "LOW"


    # Extremely high flood probability
    # or very high combined impact
    if (
        probability >= 90
        or score >= 75
    ):

        return "IMMEDIATE ACTION"


    # High risk / high impact
    if (
        probability >= 75
        or score >= 55
    ):

        return "HIGH PRIORITY"


    # Moderate flood risk
    return "WATCH"


results["action_priority"] = results.apply(
    get_action_priority,
    axis=1
)


# =========================================================
# RANK VILLAGES
# =========================================================

results = results.sort_values(
    by=[
        "priority_score",
        "flood_probability",
        "population_affected"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(
    drop=True
)


results.insert(
    0,
    "priority_rank",
    range(
        1,
        len(results) + 1
    )
)


# =========================================================
# SAVE FINAL RESULTS
# =========================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY COUNTS
# =========================================================

high = (
    results["risk_level"]
    == "HIGH"
).sum()

moderate = (
    results["risk_level"]
    == "MODERATE"
).sum()

low = (
    results["risk_level"]
    == "LOW"
).sum()


immediate = (
    results["action_priority"]
    == "IMMEDIATE ACTION"
).sum()

high_priority = (
    results["action_priority"]
    == "HIGH PRIORITY"
).sum()

watch = (
    results["action_priority"]
    == "WATCH"
).sum()


# =========================================================
# FINAL DISPLAY
# =========================================================

print("\n================================")
print(" ANALYSIS COMPLETED")
print("================================")

print(
    "Total villages:",
    len(results)
)

print("\nFlood Risk:")

print(
    "HIGH:",
    high
)

print(
    "MODERATE:",
    moderate
)

print(
    "LOW:",
    low
)


print("\nAction Priority:")

print(
    "IMMEDIATE ACTION:",
    immediate
)

print(
    "HIGH PRIORITY:",
    high_priority
)

print(
    "WATCH:",
    watch
)


# =========================================================
# TOP 20 PRIORITY VILLAGES
# =========================================================

print("\n================================")
print(" TOP 20 VILLAGES REQUIRING ACTION")
print("================================")


top20 = results.head(20)[
    [
        "priority_rank",
        "village_name",
        "block_name",
        "district",
        "flood_probability",
        "population_affected",
        "schools_affected",
        "hospitals_affected",
        "priority_score",
        "action_priority"
    ]
]


print(
    top20.to_string(
        index=False
    )
)


# =========================================================
# DISTRICT-WISE PRIORITY SUMMARY
# =========================================================

urgent = results[
    results["action_priority"].isin(
        [
            "IMMEDIATE ACTION",
            "HIGH PRIORITY"
        ]
    )
]


print("\n================================")
print(" DISTRICT-WISE PRIORITY")
print("================================")


if len(urgent) > 0:

    district_summary = (

        urgent.groupby(
            "district"
        )

        .size()

        .sort_values(
            ascending=False
        )

    )

    print(
        district_summary
    )

else:

    print(
        "No villages currently require "
        "immediate/high-priority action."
    )


print(
    "\nFull results saved to:",
    OUTPUT_FILE
)