# Milestone 2: Python modeling and evaluation

Start with **Milestone_2.html** for findings and explanations. **train.py** contains
the modeling workflow and **preprocessing.py** contains fold-fitted preprocessing.
All supplementary results and fitted model files are in **supporting/**.

Run locally: `python train.py` from this folder. The default input is the adjacent
`milestone1(Data cleaning)/supporting/handoff/` folder. To use another location:
`python train.py --data-dir "path/to/handoff"`.

Install the versions in requirements.txt in an isolated Python environment.
No GPU, cloud upload or training service is needed. Forests use two CPU workers.

## Result of this run

Ridge (alpha=100) with KNN has the lowest mean 5-fold RMSE (24.11), but the
median-imputed Ridge is essentially tied (24.12). The selected model's exploratory
test MAE is 11.69, RMSE 20.22, and R² 0.0267. The bootstrap R² interval crosses
zero; paired baseline comparisons do not establish superiority. High glucose is
substantially underpredicted. These are weak results, not a validated useful model.
The median baseline has lower MAE; the winner was chosen using predeclared RMSE.

The main script runs five shared development folds for baselines, linear
regression, Ridge, and Random Forest with median/KNN variants. The winner is
selected by mean CV RMSE before test loading. Only that winner and the two
predeclared baselines are evaluated on test data. Test rows were seen in earlier
EDA; this limitation is disclosed and external confirmation is still needed.

The input deliberately has missing predictors: the R-imputed combined CSV must
not be used for cross-validation because fold-level imputation must be fitted
again. source_row is an audit identifier, never a predictor. Model outputs use
original glucose units, not classification accuracy.

To load a fitted pipeline, keep preprocessing.py importable, then use joblib.load.
Serialized models require the same compatible package versions; load only trusted
local artifacts. Neither scripts nor report make clinical claims. The exact raw
download URL/license and team details remain to be supplied before submission.

Run `python supporting/verify_results.py` to check inputs, fold results, leakage
audits, saved predictions, model loading and evaluation outputs without retraining.
