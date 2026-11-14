"""Read-only dataset experiment; outputs diagnostic results, not cleaned data."""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiments'
OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(ROOT / 'data/raw/framingham.csv')
df = raw.dropna(subset=['glucose']).copy()
# Exclude the outcome, future CHD outcome, and diabetes of uncertain provenance.
X = df.drop(columns=['glucose', 'TenYearCHD', 'diabetes'])
targets = ['heartRate', 'BMI', 'totChol', 'cigsPerDay', 'education', 'BPMeds']
records = []
for seed in range(5):
    train, test = train_test_split(X, test_size=.25, random_state=1900 + seed)
    for target in targets:
        # Evaluate on unseen rows with known target values. Smoking intensity is
        # evaluated among smokers, matching the actual missingness in this data.
        eligible = test[target].notna()
        donors = train
        if target == 'cigsPerDay':
            eligible &= test.currentSmoker.eq(1)
            donors = train.loc[train.currentSmoker.eq(1)]
        query = test.loc[eligible].copy()
        truth = query[target].to_numpy()
        query[target] = np.nan
        distance_cols = ['male', 'age', 'currentSmoker', 'cigsPerDay',
                         'prevalentStroke', 'prevalentHyp', 'totChol',
                         'sysBP', 'diaBP', 'BMI', 'heartRate']
        if target not in distance_cols:
            distance_cols.append(target)
        donors = donors[distance_cols]
        query = query[distance_cols]
        col = distance_cols.index(target)
        categorical = target in ['education', 'BPMeds']
        observed = donors[target].dropna()
        baseline = observed.mode().iloc[0] if categorical else observed.median()
        predictions = {'mode' if categorical else 'median': np.full(len(query), baseline)}
        # StandardScaler preserves NaNs; all parameters use donor rows only.
        scaler = StandardScaler().fit(donors)
        donor_scaled = scaler.transform(donors)
        query_scaled = scaler.transform(query)
        for k in [3, 5, 10]:
            imputer = KNNImputer(n_neighbors=k, weights='uniform').fit(donor_scaled)
            estimate = imputer.transform(query_scaled)[:, col]
            predictions[f'knn_{k}'] = estimate * scaler.scale_[col] + scaler.mean_[col]
        for method, pred in predictions.items():
            row = dict(seed=seed, variable=target, method=method, n=len(truth),
                       mae=float(np.mean(np.abs(truth-pred))),
                       rmse=float(np.sqrt(np.mean((truth-pred)**2))))
            if categorical:
                # Explicit ordinal rounding / binary majority threshold. This is
                # a screening approximation, not general mixed-type neighbor voting.
                labels = np.floor(pred + .5).clip(observed.min(), observed.max())
                row['accuracy'] = float(np.mean(labels == truth))
                classes = sorted(observed.unique())
                row['balanced_accuracy'] = float(np.mean([
                    np.mean(labels[truth == c] == c) for c in classes if np.any(truth == c)]))
            records.append(row)
    print(f'Completed repetition {seed+1}/5', flush=True)
results = pd.DataFrame(records)
summary = results.groupby(['variable', 'method']).agg(
    mae=('mae', 'mean'), mae_sd=('mae', 'std'), rmse=('rmse', 'mean'),
    rmse_sd=('rmse', 'std'), accuracy=('accuracy', 'mean'),
    balanced_accuracy=('balanced_accuracy', 'mean')).reset_index()
results.to_csv(OUT / 'repeated_scores.csv', index=False)
summary.to_csv(OUT / 'summary.csv', index=False)
print(summary.round(4).to_string(index=False))
print('Observed-target missing counts:', df[targets].isna().sum().to_dict())
print('Category proportions:')
for c in ['education', 'BPMeds']:
    print(c, df[c].value_counts(normalize=True).round(4).to_dict())
