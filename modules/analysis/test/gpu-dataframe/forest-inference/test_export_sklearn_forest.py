#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regression tests: /usr/bin/python3 -B test_export_sklearn_forest.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)

from export_sklearn_forest import export_model, export_threshold


class ForestExporterTests(unittest.TestCase):
    def test_encodes_infinite_thresholds_without_accepting_nan(self):
        self.assertEqual(export_threshold(np.inf), 'Infinity')
        self.assertEqual(export_threshold(-np.inf), '-Infinity')
        self.assertEqual(export_threshold(0.5), 0.5)
        with self.assertRaisesRegex(ValueError, 'threshold.*NaN'):
            export_threshold(np.nan)

    def test_exports_missing_only_splits_as_strict_json_and_through_cli(self):
        features = np.array([[0], [1], [2], [3], [np.nan], [np.nan]], dtype=np.float32)
        queries = np.array([[0], [np.nan]], dtype=np.float32)
        for estimator in [RandomForestRegressor, RandomForestClassifier]:
            with self.subTest(estimator=estimator.__name__):
                model = estimator(n_estimators=1, bootstrap=False, random_state=0)
                model.fit(features, [0, 0, 0, 0, 1, 1])
                self.assertTrue(np.isposinf(model.estimators_[0].tree_.threshold[0]))
                np.testing.assert_array_equal(model.predict(queries), [0, 1])
                np.testing.assert_array_equal(model.apply(queries), [[1], [2]])
                exported = export_model(model)
                serialized = json.dumps(exported, allow_nan=False)
                self.assertEqual(json.loads(serialized)['trees'][0]['threshold'][0], 'Infinity')
                with tempfile.TemporaryDirectory(prefix='forest-export-test-') as directory:
                    input_path = Path(directory) / 'model.joblib'
                    output_path = Path(directory) / 'model.json'
                    joblib.dump(model, input_path)
                    result = subprocess.run(
                        [sys.executable, '-B', str(Path(__file__).with_name('export_sklearn_forest.py')),
                         str(input_path), str(output_path)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(output_path.read_text()), exported)

    def test_rejects_stratified_classifier_initializer(self):
        # Reviewer reproducer: sampling one raw score cannot represent this initializer.
        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        model = GradientBoostingClassifier(
            init=DummyClassifier(strategy='stratified', random_state=1),
            n_estimators=1, max_depth=1, random_state=0,
        ).fit(features, [0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, 'init'):
            export_model(model)

    def test_checks_fitted_classifier_strategy(self):
        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        model = GradientBoostingClassifier(
            init=DummyClassifier(strategy='stratified', random_state=1),
            n_estimators=1, max_depth=1, random_state=0,
        ).fit(features, [0, 1, 0, 1])
        # predict_proba uses the strategy saved at fit, not this public attribute.
        model.init_.strategy = 'prior'
        with self.assertRaisesRegex(ValueError, 'init'):
            export_model(model)

    def test_rejects_dummy_subclasses_with_row_dependent_predictions(self):
        class RowDependentClassifier(DummyClassifier):
            def predict_proba(self, features):
                probability = np.where(features[:, 0] < 2, 0.25, 0.75)
                return np.column_stack([1 - probability, probability])

        class RowDependentRegressor(DummyRegressor):
            def predict(self, features, return_std=False):
                return features[:, 0].astype(float)

        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        for model, targets in [
            (GradientBoostingClassifier(init=RowDependentClassifier(),
                                        n_estimators=1, random_state=0), [0, 1, 0, 1]),
            (GradientBoostingRegressor(init=RowDependentRegressor(),
                                       n_estimators=1, random_state=0), [1, 2, 4, 8]),
        ]:
            with self.subTest(estimator=type(model).__name__):
                model.fit(features, targets)
                with self.assertRaisesRegex(ValueError, 'init'):
                    export_model(model)

    def test_accepts_constant_classifier_probability_strategies(self):
        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        for strategy in ['prior', 'most_frequent', 'uniform', 'constant']:
            with self.subTest(strategy=strategy):
                initializer = DummyClassifier(strategy=strategy, constant=1)
                model = GradientBoostingClassifier(
                    init=initializer, n_estimators=1, max_depth=1, random_state=0,
                ).fit(features, [0, 0, 0, 1])
                exported = export_model(model)
                raw = model._raw_predict_init(features)
                np.testing.assert_array_equal(raw, np.repeat(raw[:1], len(features), axis=0))
                self.assertEqual(exported['base_scores'], [0.0, float(raw[0, 0])])

    def test_accepts_constant_regressor_strategies(self):
        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        for strategy in ['mean', 'median', 'quantile', 'constant']:
            with self.subTest(strategy=strategy):
                initializer = DummyRegressor(strategy=strategy, constant=3, quantile=0.25)
                model = GradientBoostingRegressor(
                    init=initializer, n_estimators=1, max_depth=1, random_state=0,
                ).fit(features, [1, 2, 4, 8])
                exported = export_model(model)
                raw = model._raw_predict_init(features)
                np.testing.assert_array_equal(raw, np.repeat(raw[:1], len(features), axis=0))
                self.assertEqual(exported['base_scores'], raw[0].tolist())

    def test_accepts_zero_initializers(self):
        features = np.array([[0], [1], [2], [3]], dtype=np.float32)
        classifier = GradientBoostingClassifier(init='zero', n_estimators=1, random_state=0)
        regressor = GradientBoostingRegressor(init='zero', n_estimators=1, random_state=0)
        self.assertEqual(export_model(classifier.fit(features, [0, 1, 0, 1]))['base_scores'], [0, 0])
        self.assertEqual(export_model(regressor.fit(features, [1, 2, 4, 8]))['base_scores'], [0])


if __name__ == '__main__':
    unittest.main()
