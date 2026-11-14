"""Independent consistency checks; does not retrain or select models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'milestone1(Data cleaning)/supporting/handoff')
args = parser.parse_args()
support = ROOT / 'supporting'
train_path = args.data_dir / 'train_unimputed.csv'
test_path = args.data_dir / 'test_unimputed.csv'
train, test = pd.read_csv(train_path), pd.read_csv(test_path)
manifest = json.loads((support / 'run_manifest.json').read_text())
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest['train_sha256']
assert hashlib.sha256(test_path.read_bytes()).hexdigest() == manifest['test_sha256']
assert len(train) == 3081 and len(test) == 771
assert not set(train.source_row) & set(test.source_row)
scores = pd.read_csv(support / 'cv_fold_scores.csv')
summary = pd.read_csv(support / 'cv_summary.csv')
assert len(scores) == 14 * 5
assert scores.groupby('candidate').fold.nunique().eq(5).all()
assert np.isfinite(scores[['MAE', 'RMSE', 'R2']]).all().all()
audit = pd.read_csv(support / 'leakage_checks.csv')
assert len(audit) == 12 * 5 and audit.donor_overlap.eq(0).all()
assert audit.heart_median_from_fold_only.all() and audit.state_unchanged_on_predict.all()
oof = pd.read_csv(support / 'out_of_fold_predictions.csv')
assert len(oof) == len(train) and not oof.isna().any().any()
assert np.allclose(oof.actual, train.glucose)
folds = pd.read_csv(support / 'cv_fold_manifest.csv')
assert set(folds.source_row) == set(train.source_row)
assert set(folds.fold) == {1, 2, 3, 4, 5}
decision = json.loads((support / 'selection_before_test.json').read_text())
assert not decision['test_loaded_during_selection']
assert decision['selected'] == summary.iloc[0].candidate
model = joblib.load(support / 'selected_pipeline.joblib')
assert set(model.named_steps['clean'].feature_names_in_) == set(FEATURES)
assert not {'glucose', 'source_row', 'diabetes', 'TenYearCHD'} & set(FEATURES)
pred = model.predict(test[FEATURES])
prep = model.named_steps['clean']
assert len(prep.fit_index_) == len(train)
filled = prep.transform(test[FEATURES])
assert not filled.isna().any().any()
for column in ['age', 'BMI', 'totChol', 'heartRate', 'cigsPerDay', 'sysBP', 'diaBP']:
    observed = test[column].notna()
    assert np.allclose(filled.loc[observed, column], test.loc[observed, column])
try:
    prep.transform(test[FEATURES].assign(glucose=test.glucose))
except ValueError:
    pass
else:
    raise AssertionError('Target column must be rejected')
saved = pd.read_csv(support / 'test_predictions.csv')
assert np.allclose(pred, saved.predicted_glucose)
assert np.allclose(test.glucose, saved.actual_glucose)
assert list(test.source_row) == list(saved.source_row)
selected = pd.read_csv(support / 'test_scores.csv').set_index('candidate').loc[decision['selected']]
assert np.isclose(selected.MAE, mean_absolute_error(test.glucose, pred))
assert np.isclose(selected.RMSE, np.sqrt(mean_squared_error(test.glucose, pred)))
assert np.isclose(selected.R2, r2_score(test.glucose, pred))
assert len(pd.read_csv(support / 'bootstrap_draws.csv')) == 2000
ci = pd.read_csv(support / 'bootstrap_95_intervals.csv')
assert ci.lower_95.le(ci.upper_95).all()
print('PASS: hashes, disjoint rows, 70 CV scores, 60 fold-preprocessing audits, OOF coverage, saved pipeline predictions, test metrics and bootstrap outputs.')
