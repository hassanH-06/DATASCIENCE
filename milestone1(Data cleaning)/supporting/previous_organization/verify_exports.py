"""Verify exported rows and that every observed predictor/target is preserved."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
raw = pd.read_csv(root / 'data/raw/framingham.csv')
raw['source_row'] = np.arange(1, len(raw) + 1)
clean = pd.read_csv(root / 'data/cleaned/framingham_cleaned.csv')
original = raw.set_index('source_row').loc[clean.source_row].reset_index()
assert len(clean) == 3852 and not clean.isna().any().any()
assert clean.source_row.is_unique
assert set(clean.source_row) == set(raw.loc[raw.glucose.notna(), 'source_row'])
assert 'diabetes' not in clean and 'TenYearCHD' not in clean
for column in set(raw.columns) & set(clean.columns):
    known = original[column].notna()
    if column == 'BPMeds':
        left = pd.to_numeric(clean.loc[known, column]).to_numpy()
    else:
        left = clean.loc[known, column].to_numpy()
    assert np.allclose(left, original.loc[known, column].to_numpy()), column
audit = pd.read_csv(root / 'outputs/imputed_cells.csv')
assert len(audit) == 198
for column in ['BMI', 'totChol', 'heartRate', 'cigsPerDay', 'education', 'BPMeds']:
    missing = original[column].isna()
    if column != 'BPMeds':
        assert (clean[f'{column}_missing'].to_numpy() == missing.astype(int).to_numpy()).all()
    else:
        assert (clean.BPMeds.eq('Unknown').to_numpy() == missing.to_numpy()).all()
    assert len(audit[audit.variable.eq(column)]) == missing.sum()
manifest = pd.read_csv(root / 'outputs/split_manifest.csv')
assert set(manifest.source_row) == set(clean.source_row)
for name, count in [('train', 3081), ('test', 771)]:
    table = pd.read_csv(root / f'data/handoff/{name}_cleaned.csv')
    assert len(table) == count
    assert set(table.source_row) == set(manifest.loc[manifest.split.eq(name), 'source_row'])
assert set(clean.BPMeds) <= {'0', '1', 'Unknown'}
assert set(clean.education) <= {1, 2, 3, 4}
assert len(clean.columns) == 20 and 'BPMeds_missing' not in clean
print('PASS: 3852 rows, 20 columns, 198 filled cells; observed values preserved; split verified.')
