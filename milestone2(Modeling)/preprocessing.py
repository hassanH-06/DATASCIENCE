"""Fold-fitted preprocessing matching the Milestone 1 decisions."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
            'BPMeds', 'prevalentStroke', 'prevalentHyp', 'totChol',
            'sysBP', 'diaBP', 'BMI', 'heartRate']
DISTANCE = [c for c in FEATURES if c not in ['education', 'BPMeds']]
MISSING = ['education', 'BMI', 'totChol', 'cigsPerDay', 'heartRate']
NUMERIC = [c for c in FEATURES if c not in ['education', 'BPMeds']]
NUMERIC += [f'{c}_missing' for c in MISSING]
CATEGORICAL = ['education', 'BPMeds']


class GlucosePreprocessor(TransformerMixin, BaseEstimator):
    """Fit medians, mode, scales, and KNN donors using X only, within a fold.

    KNN is used only for BMI and cholesterol; all other filling rules are fixed.
    No outcome, source ID, diabetes status, or future CHD is accepted as input.
    """

    def __init__(self, method='knn', n_neighbors=10):
        self.method = method
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self._check(X)
        if self.method not in ['knn', 'median']:
            raise ValueError('Unknown preprocessing method')
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.fit_index_ = X.index.to_numpy().copy()
        self.medians_ = X[DISTANCE].median()
        self.education_mode_ = float(X.education.mode().iloc[0])
        self.smoker_median_ = float(X.loc[X.currentSmoker.eq(1), 'cigsPerDay'].median())
        if self.method == 'knn':
            self.scaler_ = StandardScaler().fit(X[DISTANCE])
            scaled = self.scaler_.transform(X[DISTANCE])
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(scaled)
        return self

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != FEATURES:
            raise ValueError('Expected only the 13 explicitly approved input features')

    def transform(self, X):
        self._check(X)
        result = X.copy()
        for c in MISSING:
            result[f'{c}_missing'] = X[c].isna().astype(int)
        if self.method == 'knn' and X[['BMI', 'totChol']].isna().any().any():
            scaled = self.scaler_.transform(X[DISTANCE])
            filled = self.scaler_.inverse_transform(self.imputer_.transform(scaled))
            for c in ['BMI', 'totChol']:
                missing = X[c].isna()
                result.loc[missing, c] = filled[missing.to_numpy(), DISTANCE.index(c)]
        for c in ['BMI', 'totChol', 'heartRate']:
            result[c] = result[c].fillna(self.medians_[c])
        missing = X.cigsPerDay.isna()
        result.loc[missing, 'cigsPerDay'] = np.where(
            X.loc[missing, 'currentSmoker'].eq(0), 0., self.smoker_median_)
        result.education = result.education.fillna(self.education_mode_).astype(int).astype(str)
        result.BPMeds = X.BPMeds.map({0.: '0', 1.: '1'}).fillna('Unknown')
        if result.isna().any().any():
            raise ValueError('Preprocessing left missing values')
        return result[NUMERIC + CATEGORICAL]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC + CATEGORICAL, dtype=object)
