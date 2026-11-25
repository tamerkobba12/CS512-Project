# Drug IC50 QSAR: Data Preprocessing and Genetic-Algorithm Feature Selection

Project 1: build linear QSAR models that predict drug potency (IC50) from molecular descriptors, using a
genetic algorithm (GA) to choose which descriptors each model uses.

## Team

- Tamer Al Kobba
- Kyle Bertrand
- Mitchell Itzaina

## Status

| Part | Status |
|---|---|
| Data preprocessing (steps 1-12 of the cleaning guide) | Done |
| Genetic algorithm (feature selection and model fitting) | Done |
| Results and plots | Done (see [Results](#results)) |
| Write-up | Not started |

## Project layout

```
Project/
  README.md
  requirements.txt
  data/
    raw/drugs_Data_Original.xlsx
    processed/                  the six files made by preprocessing.py
  src/
    preprocessing.py            cleans the data and writes data/processed
    Training/
      MainMLR.py                the genetic algorithm (from the instructor's template)
      mlr.py                    multiple linear regression (the template's, unchanged)
      FromDataFileMLR.py        reads the six files in data/processed
      FromFinessFileMLR.py      fits each model, computes R2, Q2 and the fitness
      plot_results.py           metrics and plots for the best model
```

## How to run

```
pip install -r requirements.txt
cd src
python preprocessing.py         # writes the six files to data/processed
cd Training
python MainMLR.py               # runs the GA, writes a timestamped csv here and prints the best model
python plot_results.py          # metrics and plots of the best model, saved in plots/
```

Run `MainMLR.py` and `plot_results.py` from `src/Training`: the GA output file is written to the folder you run it
from, and `plot_results.py` uses the newest one there. Useful options of `plot_results.py`:
`python plot_results.py <file>.csv` (a specific output file), `--min-all 0.7` (a stricter bar for all four values),
`--no-show` (save the figures without opening windows).

The full pipeline was checked with Python 3.13, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1 and matplotlib 3.11.2.

## Data preprocessing

`src/preprocessing.py` reads `drugs_Data_Original.xlsx` (712 molecules, IC50 plus 5,270 descriptors) and
follows the cleaning guide: convert non-numeric values to NaN, remove rows with no IC50, remove columns with more than 50% NaN,
remove duplicates, remove constant and near-constant columns, IQR outliers to NaN, median imputation, standardization.
It also converts IC50 (nM) to pIC50 (`9 - log10(IC50 in nM)`), sorts by pIC50 and splits the data
(for every 4 rows: rows 1-2 to train, row 3 to validation, row 4 to test). The code uses NumPy, pandas and scikit-learn
functions on whole columns instead of Python loops wherever it can.

It creates the six files the instructor's `FromDataFileMLR.py` reads, in `data/processed/`:

| File | Content |
|---|---|
| `Train-Data.csv`, `Validation-Data.csv`, `Test-Data.csv` | the standardized descriptors (train mean 0, SD 1) |
| `Train-pIC50.csv`, `Validation-pIC50.csv`, `Test-pIC50.csv` | pIC50, one column, not scaled |

None of the six files has a header row (the instructor's format).

| Stage | Molecules | Descriptors |
|---|---|---|
| Raw | 712 | 5,270 |
| Remove molecules with no descriptors | 377 | 5,270 |
| Remove rows with missing/censored IC50 | 317 | 5,270 |
| Remove columns with more than 50% NaN | 317 | 3,818 |
| Remove duplicate rows | 294 | 3,818 |
| Remove constant / near-constant columns | 294 | 1,997 |
| Remove columns with no variation after outlier removal | 294 | 1,679 |
| Remove highly correlated columns (\|r\| > 0.95) | 294 | 751 |

Final sizes: train 148, validation 73, test 73 molecules, each with 751 descriptors.

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
6. **Highly correlated columns are removed (step 8b).** Two descriptors that move almost together carry the same information, and a
   linear model can give them huge opposite coefficients that cancel on the training molecules but give a wild prediction for a
   molecule where the two differ. Of every group of such descriptors, the first one (in column order) is kept. The correlation is
   measured on the training molecules only. The limit is `CORRELATION_LIMIT` in `preprocessing.py`: 0.95 keeps 751 descriptors
   (0.9 would keep 582).

Items 2, 3 and 6 are our choices, not in the guide. Confirm them with the instructor.

## Genetic algorithm

The GA is in `src/Training/MainMLR.py`, filled in from the instructor's template and the project slides.

| Part | What it does |
|---|---|
| Model | Multiple linear regression (`mlr.py`) on the selected descriptors, fitted on the training molecules |
| Population | 50 rows (models) of 0/1 flags over the 751 descriptors |
| First population | Every descriptor has a 1.5% chance of being selected; a row must end up with 5 to 15 descriptors, otherwise it is drawn again |
| Generations | 1,000 populations in total (the first plus 999 new ones) |
| Next population | The best two rows are copied to rows 1 and 2; the best and second best make two children by one-point crossover (rows 3 and 4); each child gets one mutation (one descriptor switched on or off); the other 46 rows are new random rows |
| Fitness (lower is better) | The mean squared error of the leave-one-out predictions of the training molecules and the predictions of the validation molecules, divided by (1 - 2p/n), where p is the number of descriptors and n is the number of molecules (221) |
| Scores of each model | R2 of training, Q2 of training (leave-one-out), R2 of validation, R2 of test, written to the output file with its descriptors and fitness |
| Final model | The lowest fitness among the models whose R2 of training, Q2, R2 of validation and R2 of test are all at least 0.6 and below 1 (0.5, the slides' bar, if no model reaches 0.6); on equal fitness, the one with the fewest descriptors |

R2 of validation and test is R2pred: the validation and test molecules are compared with the mean of the *training* pIC50.
The random seed is fixed (`np.random.seed(42)` in `main()`), so a run can be repeated; remove that line for a different run each time.

The numbers above are constants at the top of `MainMLR.py` (`MIN_FEATURES`, `MAX_FEATURES`, `PARENT_RANKS`, `NUM_CHILDREN`, `NUM_MUTATIONS`)
and in `main()` (`numOfPop`, `R2req_*`). The slides also describe a second recipe: parents are the third and fourth best rows and
children fill the other 48 rows. To use it, set `PARENT_RANKS = (2, 3)` and `NUM_CHILDREN = 48`.

### Where the code differs from the template

1. All four requirements are 0.6. The template comment asks for 0.6 / 0.5 / 0.5 and the slides ask for 0.5.
2. `PerformOneThousandIteration` sets `OldPopulation = population` after every generation. Without that line the next generation is
   always bred from the first population.
3. `cv_predict` takes the number out of the one-element prediction with `[0]` (needed on NumPy 2). `FromFinessFileMLR.py` has no other change
   apart from the `np.` prefixes, and `mlr.py` is unchanged.
4. `FromDataFileMLR.py` reads the six files from `data/processed` (the path is built from the location of the file, so it works from any folder).
5. `pickBestModel` (new) reads the output file at the end and prints the final model.
6. `plot_results.py` (new) is not part of the template.

## Results

Run with seed 42 (1,000 generations, 49,870 models scored; 1,586 different models reach 0.6 on all four values).

**Final model: 25 descriptors, fitness 0.31**

| | Training | Validation | Test |
|---|---|---|---|
| Molecules | 148 | 73 | 73 |
| R2 (R2pred for validation and test) | 0.91 | 0.86 | 0.71 |
| Q2 (leave-one-out) | 0.87 | | |
| RMSE | 0.41 | 0.46 | 0.67 |
| MAE | 0.31 | 0.38 | 0.55 |
| Pearson r | 0.95 | 0.93 | 0.85 |

All four values the instructor asked for (R2 of training, Q2, R2 of validation, R2 of test) are above 0.6.

Descriptors of the final model (0-based column positions in the processed data files, which are in the same order as the descriptors in the raw file):
18, 20, 29, 149, 219, 234, 248, 340, 351, 403, 409, 413, 522, 594, 637, 638, 639, 642, 649, 652, 661, 665, 673, 682, 749.

`plot_results.py` refits this model on the training molecules and writes, in `plots/`:

| File | Content |
|---|---|
| `1_metrics.png` | the four R2 / Q2 values against the 0.6 and 0.5 bars, and a table of R2, Q2, RMSE, MAE and r |
| `2_observed_vs_predicted.png` | observed against predicted pIC50 for the training, validation and test molecules |
| `3_ga_progress.png` | the lowest fitness found so far while the GA ran |
| `model_metrics.csv` | the metrics of the table above |
| `predictions.csv` | observed, predicted and residual pIC50 of every molecule |

### Notes

- **The validation set is part of the fitness; the test set is not.** The fitness that drives the GA uses the leave-one-out predictions of the
  training molecules and the predictions of the validation molecules, so the validation result (R2 0.86) is not an independent check.
  The test molecules only act as a pass/fail bar on the final model (R2 of test at least 0.6, or 0.5 in the slides). The template defines
  these minimums (`R2req_*`) but never uses them, so `pickBestModel` applies them to the output file. Here the bar did not change the result:
  the model with the lowest fitness in the whole output file already passes it, so the test result (R2 0.71) is not affected by the choice.

- **A few test molecules cause much of the test error.** Test molecule 69 (observed 9.0, predicted 6.83) is 15% of the squared test error;
  without it the R2 of test would be 0.74. The most potent test molecules (pIC50 of 7.5 or more) are predicted 0.48 too low on average.
