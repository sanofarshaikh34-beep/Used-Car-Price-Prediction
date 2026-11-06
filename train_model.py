import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET = "CAR DETAILS FROM CAR DEKHO.csv"

CURRENT_YEAR = 2026

df = pd.read_csv(DATASET)

df.columns = df.columns.str.strip().str.lower()

df = df.drop_duplicates()

df = df.dropna(subset=[
    "name",
    "selling_price",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
])

# Remove invalid records
df = df[
    (df["selling_price"] > 0) &
    (df["year"] >= 1990) &
    (df["year"] <= CURRENT_YEAR) &
    (df["km_driven"] >= 0)
].copy()


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Extract car brand
df["brand"] = df["name"].str.strip().str.split().str[0]

# Calculate car age
df["car_age"] = CURRENT_YEAR - df["year"]

# Avoid division by zero
df["car_age"] = df["car_age"].clip(lower=1)

# Calculate kilometres driven per year
df["km_per_year"] = (
    df["km_driven"] / df["car_age"]
)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

features = [
    "name",
    "brand",
    "year",
    "car_age",
    "km_driven",
    "km_per_year",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

X = df[features].copy()

y = df["selling_price"].copy()


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_features = [
    "year",
    "car_age",
    "km_driven",
    "km_per_year"
]

categorical_features = [
    "name",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. RANDOM FOREST PIPELINE
# ==========================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Log transformation is applied to target
rf_model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

rf_parameters = {
    "regressor__regressor__n_estimators": [300, 500, 700],
    "regressor__regressor__max_depth": [10, 15, 20, 30, None],
    "regressor__regressor__min_samples_split": [2, 5, 10],
    "regressor__regressor__min_samples_leaf": [1, 2, 3, 4],
    "regressor__regressor__max_features": [0.7, 1.0]
}


# ==========================================
# 7. EXTRA TREES PIPELINE
# ==========================================

et_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

et_model = TransformedTargetRegressor(
    regressor=et_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

et_parameters = {
    "regressor__regressor__n_estimators": [300, 500, 700],
    "regressor__regressor__max_depth": [10, 15, 20, 30, None],
    "regressor__regressor__min_samples_split": [2, 5, 10],
    "regressor__regressor__min_samples_leaf": [1, 2, 3, 4],
    "regressor__regressor__max_features": [0.7, 1.0]
}


# ==========================================
# 8. RANDOM FOREST TRAINING
# ==========================================

print("\nTraining Random Forest...")

rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_parameters,
    n_iter=15,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)


# ==========================================
# 9. EXTRA TREES TRAINING
# ==========================================

print("\nTraining Extra Trees...")

et_search = RandomizedSearchCV(
    estimator=et_model,
    param_distributions=et_parameters,
    n_iter=15,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

et_search.fit(X_train, y_train)


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print(
    "Random Forest CV R2:",
    round(rf_search.best_score_, 4)
)

print(
    "Extra Trees CV R2:",
    round(et_search.best_score_, 4)
)


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

if rf_search.best_score_ >= et_search.best_score_:

    model = rf_search.best_estimator_

    best_model_name = "Random Forest"

    best_parameters = rf_search.best_params_

else:

    model = et_search.best_estimator_

    best_model_name = "Extra Trees"

    best_parameters = et_search.best_params_


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

# Prevent negative price predictions
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": features,
        "mae": float(mae),
        "r2": float(r2),
        "model_name": best_model_name,
        "current_year": CURRENT_YEAR,
        "best_parameters": best_parameters
    },
    "car_price_model_experiment.pkl"
)


# ==========================================
# 14. FINAL RESULTS
# ==========================================

print("\n======================================")
print("       CARWISE MODEL RESULTS")
print("======================================")

print("Dataset Records:", len(df))

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print("Best Algorithm:", best_model_name)

print("MAE: ₹", round(mae, 2))

print("R2 Score:", round(r2, 4))

print("R2 Percentage:", round(r2 * 100, 2), "%")

print("\nBest Parameters:")

print(best_parameters)

print("\nModel saved successfully!")

print("File: car_price_model.pkl")

print("======================================")