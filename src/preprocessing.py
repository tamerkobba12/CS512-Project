import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ============================================================
# 1. LOAD THE DATA
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
file_name = ROOT / "data" / "raw" / "drugs_Data_Original.xlsx"

# CHANGED: the data is an Excel file ("na" is read directly as NaN).
data = pd.read_excel(file_name, engine="calamine", na_values=["na"])

# ADDED: "No." and "NAME" are identifiers, not descriptors.
data = data.drop(columns=["No.", "NAME"])

# The first column is Y (IC50).
# All remaining columns are X variables.
print("Original shape:", data.shape)


# ============================================================
# 2. CONVERT NON-NUMERIC VALUES TO NaN
# ============================================================

data = data.apply(pd.to_numeric, errors="coerce")

print("\nTotal NaN values:", data.isna().sum().sum())


# ============================================================
# 2b. ADDED: REMOVE MOLECULES THAT HAVE NO DESCRIPTORS
# ============================================================

# About half of the molecules have no descriptors at all (every X is NaN).
# Without this step, step 4 would remove almost every X column.

no_descriptors = data.iloc[:, 1:].isna().mean(axis=1) > 0.99

data = data[~no_descriptors]

print("\nAfter removing molecules without descriptors:")
print("Shape:", data.shape)


# ============================================================
# 3. REMOVE ROWS WHERE Y (IC50) IS NaN
# ============================================================

target_column = data.columns[0]

data = data.dropna(subset=[target_column])

print("\nAfter removing rows with missing IC50:")
print("Shape:", data.shape)


# ============================================================
# 4. REMOVE X COLUMNS WITH MORE THAN 50% NaN
# ============================================================

x_columns = data.columns[1:]

nan_fraction = data[x_columns].isna().mean()

columns_to_remove = nan_fraction[nan_fraction > 0.50].index

data = data.drop(columns=columns_to_remove)

print("\nAfter removing columns with >50% NaN:")
print("Shape:", data.shape)


# ============================================================
# 5. REMOVE DUPLICATE ROWS
# ============================================================

number_before = len(data)

data = data.drop_duplicates()

number_after = len(data)

print("\nDuplicate rows removed:",
      number_before - number_after)

print("Shape:", data.shape)


# ============================================================
# 6. REMOVE CONSTANT AND NEAR-CONSTANT X COLUMNS
# ============================================================

Y = data.iloc[:, 0].copy()
X = data.iloc[:, 1:].copy()


# ------------------------------------------------------------
# 6a. Remove CONSTANT columns
# ------------------------------------------------------------

constant_columns = [ ]

for column in X.columns:
    number_unique = X[column].nunique(dropna=True)
    if number_unique <= 1:
        constant_columns.append(column)

print("\nConstant X columns removed:", len(constant_columns))

X = X.drop(columns=constant_columns)


# ------------------------------------------------------------
# 6b. Remove NEAR-CONSTANT columns
# ------------------------------------------------------------

near_constant_threshold = 0.99

near_constant_columns = [ ]

for column in X.columns:
    # Calculate frequency of each value.
    frequencies = X[column].value_counts(
        normalize=True,
        dropna=True
    )

    # Check the most common value.
    if len(frequencies) > 0:

        highest_frequency = frequencies.iloc[0]

        if highest_frequency >= near_constant_threshold:
            near_constant_columns.append(column)

print("Near-constant X columns removed:", len(near_constant_columns))

X = X.drop(columns=near_constant_columns)

print("\nNumber of X variables remaining:", X.shape[1])


# ============================================================
# 10.IC50 (nM) -> pIC50   (Y is NOT scaled)
# ============================================================

# pIC50 = -log10(IC50 in mol/L) = 9 - log10(IC50 in nM)
Y = (9 - np.log10(Y)).rename("pIC50")


# ============================================================
# 11. SORT BY pIC50 AND SPLIT INTO TRAIN / VALID / TEST
# ============================================================



data_sorted = pd.concat([Y, X], axis=1).sort_values("pIC50", kind="mergesort")

# For every 4 rows: rows 1-2 -> train, row 3 -> validation, row 4 -> test.
position = np.arange(len(data_sorted)) % 4

train = data_sorted[position < 2]
valid = data_sorted[position == 2]
test = data_sorted[position == 3]

Y_train, X_train = train.iloc[:, 0], train.iloc[:, 1:].copy()
Y_valid, X_valid = valid.iloc[:, 0], valid.iloc[:, 1:].copy()
Y_test, X_test = test.iloc[:, 0], test.iloc[:, 1:].copy()

print("\nSplit sizes: train =", len(train),
      "| validation =", len(valid),
      "| test =", len(test))


# ============================================================
# 7. DETECT OUTLIERS IN EACH X AND MAKE THEM NaN
# ============================================================



for column in X_train.columns:

    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    for X_part in (X_train, X_valid, X_test):

        outliers = (
            (X_part[column] < lower_limit) |
            (X_part[column] > upper_limit)
        )

        X_part.loc[outliers, column] = np.nan

print("\nOutliers in X have been converted to NaN.")


# ============================================================
# 8. IMPUTE REMAINING NaN VALUES IN X
# ============================================================


imputer = SimpleImputer(strategy="median")

imputer.fit(X_train)

X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns, index=X_valid.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)


constant_after_imputation = X_train.columns[X_train.nunique() <= 1]

print("Columns with no variation left (removed):", len(constant_after_imputation))

X_train = X_train.drop(columns=constant_after_imputation)
X_valid = X_valid.drop(columns=constant_after_imputation)
X_test = X_test.drop(columns=constant_after_imputation)


# ============================================================
# 9. SCALE THE X VARIABLES
# ============================================================


scaler = StandardScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns, index=X_valid.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print("\nX variables have been standardized.")


# ============================================================
# COMBINE pIC50 AND CLEANED/SCALED X  (train / valid / test)
# ============================================================

train = pd.concat([Y_train, X_train], axis=1)
valid = pd.concat([Y_valid, X_valid], axis=1)
test = pd.concat([Y_test, X_test], axis=1)


# ============================================================
# CHECK THE FINAL DATASETS
# ============================================================

print("\n========================================")
print("FINAL DATASETS")
print("========================================")

print("train:", train.shape)
print("valid:", valid.shape)
print("test: ", test.shape)

print("\nRemaining NaN values:",
      train.isna().sum().sum() + valid.isna().sum().sum() + test.isna().sum().sum())


# ============================================================
# 12.CREATE THE 6 FILES
# ============================================================

out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)

X_train.to_csv(out / "TrainX.csv", index=False)
Y_train.to_frame().to_csv(out / "TrainY.csv", index=False)

X_valid.to_csv(out / "ValidX.csv", index=False)
Y_valid.to_frame().to_csv(out / "ValidY.csv", index=False)

X_test.to_csv(out / "TestX.csv", index=False)
Y_test.to_frame().to_csv(out / "TestY.csv", index=False)
print("\nSaved: TrainX.csv, TrainY.csv, ValidX.csv, ValidY.csv, TestX.csv, TestY.csv")