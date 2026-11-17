# Drug IC50 QSAR: Data Preprocessing and Genetic-Algorithm Feature Selection

Project 1: build linear QSAR models that predict drug potency (IC50) from molecular descriptors, using a
genetic algorithm (GA) to choose which descriptors each model uses.

## Team

- Tamer Al Kobba
- Kyle Bertrand
- Mitchell Itzaina
- Darian White

## Status

| Part | Status |
|---|---|
| Data preprocessing (steps 1-12 of the cleaning guide) | Done |
| Genetic algorithm (feature selection and model fitting) | Not started |
| Results and write-up | Not started |

## What's done: data preprocessing

`data_preprocessing.py` reads `drugs_Data_Original.xlsx` (712 molecules, IC50 plus 5,270 descriptors) and
follows the cleaning guide: convert non-numeric values to NaN, remove rows with no IC50, remove columns with more than 50% NaN,
remove duplicates, remove constant and near-constant columns, IQR outliers to NaN, median imputation, standardization.
It also converts IC50 (nM) to pIC50 (`9 - log10(IC50 in nM)`), sorts by pIC50 and splits the data
(for every 4 rows: rows 1-2 to train, row 3 to validation, row 4 to test).

It creates six files: `TrainX.csv`, `TrainY.csv`, `ValidX.csv`, `ValidY.csv`, `TestX.csv`, `TestY.csv`.
Each Y file has one column (`pIC50`). X values are standardized (train mean 0, SD 1), and Y is not scaled.

| Stage | Molecules | Descriptors |
|---|---|---|
| Raw | 712 | 5,270 |
| Remove molecules with no descriptors | 377 | 5,270 |
| Remove rows with missing/censored IC50 | 317 | 5,270 |
| Remove columns with more than 50% NaN | 317 | 3,818 |
| Remove duplicate rows | 294 | 3,818 |
| Remove constant / near-constant columns | 294 | 1,997 |
| Remove columns with no variation after outlier removal | 294 | 1,679 |

Final sizes: train 148, validation 73, test 73 molecules, each with 1,679 descriptors.

### Where the code differs from the cleaning guide

Marked `ADDED` / `CHANGED` in the code.

1. **Excel input and ID columns.** The data is an `.xlsx`; "No." and "NAME" are identifiers, so they are dropped.
2. **Molecules with no descriptors are removed first (step 2b).** 335 of the 712 molecules have every descriptor missing.
   Left in, step 4 would delete 5,269 of the 5,270 descriptors (about half of every column is NaN because of those rows).
3. **The split happens before steps 7-9.** The outlier limits, medians, means and SDs are calculated on the training rows only, then
   applied to validation and test, so no information from them leaks into the preprocessing (as the guide recommends).
4. **Columns with no variation after imputation are removed.** After outliers become NaN and are imputed, 318 columns are constant in the
   training data. They cannot be scaled and would break the regression.
5. Fixed an indentation error in step 6b of the guide's sample code.

Items 2 and 3 are our choices, not in the guide. Confirm them with the instructor.


## What's left: genetic algorithm

From the project slides:

- [ ] **Initial population:** 50 models. Each is a row of 0/1 flags over the 1,679 descriptors, and each descriptor has a 1.5% chance of being selected (about 25 per model).
- [ ] **Fit each model:** Linear regression on the selected descriptors (`a*x2 + b*x5 + ... + e = Y`) using the training data, then predict train, validation and test.
- [ ] **Fitness for each model:** R2 and Q2 on training, R2 on validation and R2 on test, plus RMSE. One fitness value per model (array of 50).
- [ ] **Next generation:** Keep the two best rows, create children with 1-point / 2-point / n-point crossover, apply mutation, and fill the rest with more children or random rows.
- [ ] **Iterate** over several generations.
- [ ] **Select the best model:** lowest fitness, with R2 (train), Q2 (train), R2 (validation) and R2 (test) all at least 0.5 and below 1.
- [ ] **Results and write-up.**