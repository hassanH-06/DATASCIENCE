"""Reproduce Milestone 2 locally: python train.py [--data-dir PATH]."""
import argparse
import hashlib
import html
import json
import platform
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import GlucosePreprocessor, FEATURES, NUMERIC, CATEGORICAL

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / 'supporting'
FIG = SUPPORT / 'figures'


def metrics(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred))}


def pipeline(method, model):
    encoding = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('category', OneHotEncoder(categories=[['1', '2', '3', '4'], ['0', '1', 'Unknown']],
                                  drop='first', handle_unknown='error', sparse_output=False), CATEGORICAL)
    ], verbose_feature_names_out=False)
    return Pipeline([('clean', GlucosePreprocessor(method=method)),
                     ('encode', encoding), ('model', model)])


def candidates():
    # This finite menu and selection rule are fixed before test-file loading.
    menu = {'Mean baseline': DummyRegressor(strategy='mean'),
            'Median baseline': DummyRegressor(strategy='median')}
    for method in ['median', 'knn']:
        menu[f'Linear | {method}'] = pipeline(method, LinearRegression())
        for alpha in [1., 10., 100.]:
            menu[f'Ridge alpha={alpha:g} | {method}'] = pipeline(method, Ridge(alpha=alpha))
        for depth, leaf in [(6, 5), (None, 5)]:
            label = 'unlimited' if depth is None else str(depth)
            menu[f'Random Forest depth={label} leaf={leaf} | {method}'] = pipeline(
                method, RandomForestRegressor(n_estimators=150, max_depth=depth,
                    min_samples_leaf=leaf, max_features=1., random_state=262, n_jobs=2))
    return menu


def plot_save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=140)
    plt.close()


def bootstrap(y, prediction, baseline, repetitions=2000):
    rng = np.random.default_rng(262)
    rows = []
    for b in range(repetitions):
        ids = rng.integers(0, len(y), len(y))
        actual, pred, base = y[ids], prediction[ids], baseline[ids]
        values, base_values = metrics(actual, pred), metrics(actual, base)
        row = {'draw': b, **values}
        row.update({f'delta_{key}_vs_mean': values[key] - base_values[key] for key in values})
        rows.append(row)
    draws = pd.DataFrame(rows)
    ci = pd.DataFrame([{'metric': c, 'lower_95': draws[c].quantile(.025),
                        'upper_95': draws[c].quantile(.975)} for c in draws if c != 'draw'])
    return draws, ci


def table(frame):
    return frame.to_html(index=False, float_format=lambda v: f'{v:.4f}', border=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'milestone1(Data cleaning)' / 'supporting/handoff')
    args = parser.parse_args()
    for folder in [SUPPORT, FIG]:
        folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train_path = args.data_dir / 'train_unimputed.csv'
    test_path = args.data_dir / 'test_unimputed.csv'
    train = pd.read_csv(train_path)
    assert train.source_row.is_unique and train.glucose.notna().all()
    assert 'diabetes' not in train and 'TenYearCHD' not in train
    X, y = train[FEATURES], train.glucose.to_numpy()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=262).split(X))
    fold_labels = np.full(len(train), -1)
    for fold, (fit_ids, validation_ids) in enumerate(folds, 1):
        assert not set(fit_ids) & set(validation_ids)
        fold_labels[validation_ids] = fold
    pd.DataFrame({'source_row': train.source_row, 'fold': fold_labels}).to_csv(SUPPORT / 'cv_fold_manifest.csv', index=False)
    menu = candidates()
    scores, oof = [], {}
    audits = []
    for name, specification in menu.items():
        oof[name] = np.full(len(y), np.nan)
        for fold, (fit_ids, validation_ids) in enumerate(folds, 1):
            fitted = clone(specification).fit(X.iloc[fit_ids], y[fit_ids])
            pred = fitted.predict(X.iloc[validation_ids])
            assert np.isfinite(pred).all()
            oof[name][validation_ids] = pred
            scores.append({'candidate': name, 'fold': fold, **metrics(y[validation_ids], pred)})
            if isinstance(fitted, Pipeline):
                prep = fitted.named_steps['clean']
                assert set(prep.fit_index_) == set(X.iloc[fit_ids].index)
                assert not set(prep.fit_index_) & set(X.iloc[validation_ids].index)
                expected = X.iloc[fit_ids].heartRate.median()
                assert prep.medians_['heartRate'] == expected
                if prep.method == 'knn':
                    assert prep.imputer_._fit_X.shape[0] == len(fit_ids)
                # Transform/predict must not update fitted state.
                state_before = joblib.hash(prep)
                fitted.predict(X.iloc[validation_ids[:3]])
                assert state_before == joblib.hash(prep)
                audits.append({'candidate': name, 'fold': fold, 'fit_rows': len(fit_ids),
                               'validation_rows': len(validation_ids), 'donor_overlap': 0,
                               'heart_median_from_fold_only': True, 'state_unchanged_on_predict': True})
        print(f'CV complete: {name}', flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(SUPPORT / 'cv_fold_scores.csv', index=False)
    summary = scores.groupby('candidate').agg(
        MAE_mean=('MAE', 'mean'), MAE_sd=('MAE', 'std'),
        RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
        R2_mean=('R2', 'mean'), R2_sd=('R2', 'std')).reset_index()
    summary = summary.sort_values(['RMSE_mean', 'MAE_mean', 'candidate']).reset_index(drop=True)
    summary.to_csv(SUPPORT / 'cv_summary.csv', index=False)
    selected = summary.iloc[0].candidate
    best_learned = summary.loc[~summary.candidate.str.contains('baseline')].iloc[0].candidate
    # Persist the frozen decision before loading any test values.
    decision = {'primary_metric': 'mean validation-fold RMSE',
                'tie_breakers': ['mean MAE', 'candidate name'],
                'selected': selected, 'best_learned_candidate': best_learned,
                'folds': 5, 'seed': 262, 'n_training': len(train),
                'test_loaded_during_selection': False}
    (SUPPORT / 'selection_before_test.json').write_text(json.dumps(decision, indent=2), encoding='utf8')
    assert all(np.isfinite(v).all() for v in oof.values())
    oof_table = pd.DataFrame({'source_row': train.source_row, 'actual': y, **oof})
    oof_table.to_csv(SUPPORT / 'out_of_fold_predictions.csv', index=False)
    pd.DataFrame(audits).to_csv(SUPPORT / 'leakage_checks.csv', index=False)

    final = clone(menu[selected]).fit(X, y)
    learned = final if selected == best_learned else clone(menu[best_learned]).fit(X, y)
    mean_model = DummyRegressor(strategy='mean').fit(X, y)
    median_model = DummyRegressor(strategy='median').fit(X, y)
    fitted_state = joblib.hash(final)
    # Test evaluation begins only after the menu, winner, and final fit are frozen.
    test = pd.read_csv(test_path)
    assert test.source_row.is_unique and test.glucose.notna().all()
    assert not set(train.source_row) & set(test.source_row)
    assert len(train) == 3081 and len(test) == 771
    Xt, yt = test[FEATURES], test.glucose.to_numpy()
    prediction = final.predict(Xt)
    assert fitted_state == joblib.hash(final)
    mean_pred = mean_model.predict(Xt)
    evaluation = {selected: prediction, 'Mean baseline': mean_pred,
                  'Median baseline': median_model.predict(Xt)}
    test_scores = pd.DataFrame([{'candidate': name, **metrics(yt, pred)} for name, pred in evaluation.items()])
    test_scores.to_csv(SUPPORT / 'test_scores.csv', index=False)
    pred_table = pd.DataFrame({'source_row': test.source_row, 'actual_glucose': yt,
                              'predicted_glucose': prediction, 'mean_baseline': mean_pred,
                              'residual_actual_minus_predicted': yt-prediction,
                              'absolute_error': np.abs(yt-prediction)})
    pred_table.to_csv(SUPPORT / 'test_predictions.csv', index=False)
    pred_table.nlargest(20, 'absolute_error').to_csv(SUPPORT / 'largest_test_errors.csv', index=False)
    draws, ci = bootstrap(yt, prediction, mean_pred)
    draws.to_csv(SUPPORT / 'bootstrap_draws.csv', index=False)
    ci.to_csv(SUPPORT / 'bootstrap_95_intervals.csv', index=False)
    # Descriptive error slices, fixed before examining test errors.
    high_cutoff = float(np.quantile(y, .95))
    groups = {'recorded female': test.male.eq(0), 'recorded male': test.male.eq(1),
              'age below 50': test.age.lt(50), 'age 50 or above': test.age.ge(50),
              'glucose above training 95th percentile': test.glucose.gt(high_cutoff),
              'glucose at/below training 95th percentile': test.glucose.le(high_cutoff)}
    subgroup = pd.DataFrame([{'group': name, 'n': int(mask.sum()),
                  'MAE': float(np.mean(np.abs(yt[mask]-prediction[mask]))),
                  'RMSE': float(np.sqrt(np.mean((yt[mask]-prediction[mask])**2))),
                  'mean_residual': float(np.mean(yt[mask]-prediction[mask]))}
                 for name, mask in groups.items() if mask.any()])
    subgroup.to_csv(SUPPORT / 'subgroup_errors.csv', index=False)
    train_score = metrics(y, final.predict(X))
    oof_score = metrics(y, oof[selected])
    # Interpretation uses development OOF/permutation data, not test-guided selection.
    if isinstance(learned, Pipeline) and isinstance(learned.named_steps['model'], (LinearRegression, Ridge)):
        names = learned.named_steps['encode'].get_feature_names_out()
        coefficients = pd.DataFrame({'feature': names, 'coefficient': learned.named_steps['model'].coef_})
        coefficients['abs_coefficient'] = coefficients.coefficient.abs()
        interpretation = coefficients.sort_values('abs_coefficient', ascending=False)
        interpretation.to_csv(SUPPORT / 'linear_coefficients.csv', index=False)
        explanation = 'Linear coefficients describe adjusted associations, not causal effects. Numerical inputs are standardized: their coefficients are per training standard deviation; category coefficients are relative to education level 1 and no BP medication. Small missingness groups and correlated inputs make coefficients unstable.'
    else:
        pi = permutation_importance(learned, X, y, n_repeats=5, random_state=262,
                                    scoring='neg_mean_absolute_error', n_jobs=1)
        interpretation = pd.DataFrame({'feature': FEATURES, 'increase_in_training_MAE': pi.importances_mean,
                                       'sd': pi.importances_std}).sort_values('increase_in_training_MAE', ascending=False)
        interpretation.to_csv(SUPPORT / 'training_permutation_importance.csv', index=False)
        explanation = 'Permutation importance is computed on training data for descriptive interpretation only. It may be optimistic and can divide credit between correlated predictors. It is not causal importance or evidence of validated feature selection.'
    if isinstance(final, DummyRegressor):
        explanation = 'The selected model is a constant baseline: it uses no feature relationships. The table below describes the best learned candidate, which did not beat the baseline under the chosen CV criterion. ' + explanation
    joblib.dump(final, SUPPORT / 'selected_pipeline.joblib')
    reloaded = joblib.load(SUPPORT / 'selected_pipeline.joblib')
    assert np.allclose(reloaded.predict(Xt), prediction)
    fingerprint = {'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
                   'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(),
                   'python': platform.python_version(), 'sklearn': sklearn.__version__,
                   'numpy': np.__version__, 'pandas': pd.__version__,
                   'elapsed_seconds': time.time()-start}
    (SUPPORT / 'run_manifest.json').write_text(json.dumps(fingerprint, indent=2), encoding='utf8')
    plt.figure(figsize=(10, 7))
    plt.barh(summary.candidate[::-1], summary.RMSE_mean[::-1], xerr=summary.RMSE_sd[::-1], color='steelblue')
    plt.xlabel('Mean 5-fold RMSE; bars show fold SD, not a confidence interval')
    plot_save('cv_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(yt, prediction, s=15, alpha=.5)
    limits = [min(yt.min(), prediction.min()), max(yt.max(), prediction.max())]
    axes[0].plot(limits, limits, 'k--'); axes[0].set(xlabel='Actual glucose', ylabel='Predicted glucose')
    axes[1].scatter(prediction, yt-prediction, s=15, alpha=.5)
    axes[1].axhline(0, color='black', linestyle='--'); axes[1].set(xlabel='Predicted glucose', ylabel='Actual minus predicted')
    plot_save('test_errors.png')
    plt.figure(figsize=(7, 4)); plt.hist(yt-prediction, bins=35, color='steelblue')
    plt.xlabel('Residual: actual minus predicted'); plt.ylabel('Test observations')
    plot_save('residual_distribution.png')
    selected_test = metrics(yt, prediction)
    test_r2_ci = ci.loc[ci.metric.eq('R2')].iloc[0]
    test_delta_ci = ci.loc[ci.metric.eq('delta_RMSE_vs_mean')].iloc[0]
    practical_summary = (
        f'The selected model has test MAE {selected_test["MAE"]:.2f}, RMSE '
        f'{selected_test["RMSE"]:.2f}, and R² {selected_test["R2"]:.4f}. '
        f'The R² interval is [{test_r2_ci.lower_95:.4f}, {test_r2_ci.upper_95:.4f}]. '
        'The improvement is small: most variation remains unexplained. '
        f'The paired RMSE difference interval versus the mean baseline is '
        f'[{test_delta_ci.lower_95:.4f}, {test_delta_ci.upper_95:.4f}]; '
        'an interval crossing zero does not establish superiority. '
        'The median baseline has lower MAE than the selected model in this run, '
        'illustrating that the preferred choice depends on the predeclared metric. '
        'KNN versus median preprocessing differs negligibly in CV RMSE; '
        'this experiment does not justify a strong claim that KNN is better. '
        'This is an exploratory comparison, not evidence of accurate clinical prediction.'
    )
    conclusion = ('The selected learned model improves mean cross-validation RMSE over the mean baseline.'
                  if not isinstance(final, DummyRegressor) else
                  'No tested learned model improved the primary mean cross-validation RMSE over the selected constant baseline. The defensible result is limited predictive signal, not a claim that a complex model is successful.')
    sections = [
        ('Objective and connection to Milestone 1', '<p>Predict measured glucose from 13 permitted demographic, behavioral and examination inputs. This is supervised regression. The original CSV has 4,240 rows; 388 missing targets were excluded in R, leaving 3,852. The supplied R split is 3,081 development and 771 test rows. We start from the Milestone 1 pre-imputation handoff to refit preprocessing within folds. No clinical utility is claimed.</p>'),
        ('Validation and preprocessing', '<p>Five shuffled K-fold splits, random seed 262, shared by every candidate. All medians, education mode, smoker median, KNN donor values, distance scales, and final linear-model scales are fitted only on each fold\'s training rows. BMI/cholesterol compare median versus standardized KNN-10. Heart rate uses median; cigarettes use smoker median; education uses mode; missing medication becomes Unknown. Five missingness flags are regenerated per fold. Fixed documented category levels use k-1 encoding; medication Unknown already represents missingness. Glucose, source_row, diabetes, and TenYearCHD never enter the features.</p><p>We use fixed candidate settings instead of a large search. Ridge alpha is compared at 1, 10, 100; Random Forest uses 150 trees, leaf size 5, and depth 6 or unlimited. No target transformation or engineered-feature search is added. Selection uses lowest mean fold RMSE, then MAE; this rule was saved before test-file loading. Selected CV scores can be optimistic after selection and are not independent final estimates.</p>'),
        ('Earlier exploration: important limitation', '<p><strong>The reserved test rows were previously exposed to EDA and exploratory imputation screening in Milestone 1. They are not a pristine independent confirmation set.</strong> This modeling run does not use them to fit preprocessing, compare candidates, tune settings, or select a winner. The prior exposure cannot be undone; external validation is needed for an independent claim.</p>'),
        ('Baseline and regression metrics', '<p>The mean and median baselines ignore the inputs and predict the corresponding development-fold target constant. These are benchmarks, not clinical models. RMSE is primary because large glucose errors matter; MAE reports typical absolute error in original glucose units. R² measures squared-error improvement relative to the evaluation sample mean and can be negative. The fitted mean baseline comes from training and can itself have a negative held-out R². Classification accuracy and its binomial interval are inappropriate for continuous glucose.</p>'),
        ('Cross-validation results', table(summary) + f'<p>Selected: <strong>{html.escape(selected)}</strong>. Best learned candidate: {html.escape(best_learned)}. Fold SD is descriptive variation, not a 95% CI; folds share training observations. {conclusion}</p><img src="supporting/figures/cv_comparison.png" alt="CV RMSE comparison">'),
        ('Final exploratory-holdout evaluation', table(test_scores) + '<p>Only the frozen winner and the two predeclared baselines are reported on test data. No other candidate is selected using these results. Training and pooled OOF scores for the winner follow; training error is not a future-performance estimate.</p>' + table(pd.DataFrame([{'sample': 'training (optimistic)', **train_score}, {'sample': 'pooled out-of-fold', **oof_score}]))),
        ('Bootstrap 95% uncertainty intervals', table(ci) + '<p>2,000 paired resamples of test observations and their fixed predictions, seed 262; percentile intervals. Deltas are selected-model metric minus training-mean-baseline metric, using the same resampled observations. Negative MAE/RMSE deltas and positive R² deltas favor the selected model. These are approximate sampling intervals conditional on the fitted model and this exploratory sample; they do not include refitting, tuning, training-sample, or prior-selection uncertainty. They assume independent representative observations; the supplied data do not establish that assumption.</p>'),
        ('Error analysis and possible bias', table(subgroup) + f'<p>The high-glucose slice uses the development-target 95th percentile ({high_cutoff:.2f}), not a diagnostic threshold. Slices are descriptive and may be small; differences do not prove discrimination or fairness. Extreme errors can dominate RMSE and are retained. No changes were made after inspecting test errors.</p><img src="supporting/figures/test_errors.png" alt="Prediction and residual diagnostics"><img src="supporting/figures/residual_distribution.png" alt="Residual distribution">'),
        ('Interpretation', '<p>' + explanation + '</p>' + table(interpretation.head(15))),
        ('Conclusions, assumptions and next questions', '<p>' + practical_summary + '</p><p>' + conclusion + ' Excluding diabetes avoids potential circularity but removes strong apparent signal. Weak linear associations, rare high-glucose cases, historical/geographic sampling, unknown fasting/treatment context, imputation assumptions, missing-target selection, uncertain provenance and license, and earlier full-data exploration limit conclusions. Source rows are not patient IDs; repeated-person dependence cannot be excluded from the supplied schema. Any future feature engineering or tuning must use development folds; these test results must not become a tuning target. An external sample is needed to establish transportability.</p>'),
        ('Reproduction and lecture alignment', '<p>Run <code>python train.py</code> from this folder with the adjacent Milestone 1 folder present. Exact dependency versions, input hashes, fold membership, fold scores, leakage checks, predictions, bootstrap draws and serialized pipeline are under supporting/. Lecture 5 slides 35–46 cover regression metrics and R² intervals; slides 67–74 cover holdout and cross-validation; slides 75–78 distinguish bootstrap uncertainty from model selection. Lecture 4 slides 80–81 introduce regularization and Ridge. Bootstrap intervals here extend the lecture\'s fixed-prediction resampling approach to regression. Adjusted training R² is not used to rank a mixture of linear models, forests and baselines; unseen-data metrics are appropriate for that comparison.</p>')
    ]
    body = '<h1>Milestone 2: Glucose Modeling and Evaluation</h1>'
    markdown = '# Milestone 2: Glucose Modeling and Evaluation\n\n'
    for title, content in sections:
        body += '<h2>' + title + '</h2>' + content
        markdown += '## ' + title + '\n\n' + content + '\n\n'
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Milestone 2</title><style>body{font:16px/1.5 sans-serif;max-width:1100px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccc;text-align:left}img{max-width:100%}h2{margin-top:32px}</style>' + body + '</html>'
    (ROOT / 'Milestone_2.html').write_text(doc, encoding='utf8')
    (ROOT / 'Milestone_2.md').write_text(markdown, encoding='utf8')
    print('\nCROSS VALIDATION\n', summary.to_string(index=False), flush=True)
    print('\nTEST\n', test_scores.to_string(index=False), flush=True)
    print('\n95% INTERVALS\n', ci.to_string(index=False), flush=True)
    print('Selected:', selected, '| Seconds:', round(time.time()-start, 1), flush=True)


if __name__ == '__main__':
    main()
