import pandas as pd
import geopandas as gpd


# =========================================================
# LOAD IMPACT DATA ONLY ONCE
# =========================================================

print("Loading GIS impact datasets...")

population = pd.read_csv(
    "data/synthetic_population_100m_8_bihar_districts.csv"
)

population_gdf = gpd.GeoDataFrame(
    population,
    geometry=gpd.points_from_xy(
        population["longitude"],
        population["latitude"]
    ),
    crs="EPSG:4326"
)


places = gpd.read_file(
    "data/places.geojson"
)

places = places.to_crs("EPSG:4326")


# =========================================================
# IMPACT ANALYSIS FUNCTION
# =========================================================

def analyze_impact(village_geometry):

    # -----------------------------------------
    # Population inside village
    # -----------------------------------------

    affected_population = population_gdf[
        population_gdf.geometry.intersects(village_geometry)
    ]

    total_population = int(
        affected_population["population"].sum()
    )


    # -----------------------------------------
    # Places inside village
    # -----------------------------------------

    affected_places = places[
        places.geometry.intersects(village_geometry)
    ]


    total_places = len(affected_places)


    # -----------------------------------------
    # Critical infrastructure
    # -----------------------------------------

    schools = 0
    hospitals = 0
    markets = 0


    if "categories" in affected_places.columns:

        categories = (
            affected_places["categories"]
            .astype(str)
            .str.lower()
        )

        schools = int(
            categories.str.contains(
                "school|education|college|university",
                regex=True
            ).sum()
        )

        hospitals = int(
            categories.str.contains(
                "hospital|clinic|health|diagnostic",
                regex=True
            ).sum()
        )

        markets = int(
            categories.str.contains(
                "market|mall|shopping",
                regex=True
            ).sum()
        )


    return {
        "population_affected": total_population,
        "total_places_affected": total_places,
        "schools_affected": schools,
        "hospitals_affected": hospitals,
        "markets_affected": markets
    }