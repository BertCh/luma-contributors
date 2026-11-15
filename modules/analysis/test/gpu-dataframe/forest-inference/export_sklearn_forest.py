#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Export trusted fitted sklearn forests, or regenerate the synthetic oracle fixture.

/usr/bin/python3 export_sklearn_forest.py --fixtures
/usr/bin/python3 export_sklearn_forest.py model.joblib model.json

Only load joblib files you trust: deserialization executes Python code. No pickle is
read by the TypeScript importer. The versioned JSON preserves sklearn's double
thresholds and leaf values; it contains no training rows or bundled dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)


def export_threshold(value):
    """Version-1 JSON encodes infinite thresholds explicitly, never as null or bare Infinity."""
    if np.isposinf(value):
        return 'Infinity'
    if np.isneginf(value):
        return '-Infinity'
    if not np.isfinite(value):
        raise ValueError('tree threshold must not be NaN')
    return float(value)


def export_model(model):
    supported = (RandomForestRegressor, RandomForestClassifier,
                 GradientBoostingRegressor, GradientBoostingClassifier)
    if type(model) not in supported:
        raise ValueError('estimator must be a supported RandomForest or GradientBoosting class')
    if getattr(model, 'n_outputs_', 1) != 1:
        raise ValueError('only single-target models are supported')
    classification = isinstance(model, (RandomForestClassifier, GradientBoostingClassifier))
    boosted = isinstance(model, (GradientBoostingRegressor, GradientBoostingClassifier))
    result = {'format': 'sklearn-forest', 'version': 1,
              'sklearn_version': sklearn.__version__,
              'estimator': type(model).__name__, 'n_features_in': int(model.n_features_in_)}
    if classification:
        result['classes'] = model.classes_.tolist()
    if boosted:
        initializer = model.init_
        zero_initializer = type(initializer) is str and initializer == 'zero'
        # Boosting reads probabilities, so uniform is constant even though predict()
        # samples labels. Stratified predict_proba() samples one-hot rows instead.
        # Exact types exclude subclasses that override prediction with a row-dependent rule.
        constant_initializer = (
            classification and type(initializer) is DummyClassifier
            and initializer.strategy in {'prior', 'most_frequent', 'uniform', 'constant'}
            and getattr(initializer, '_strategy', initializer.strategy)
            in {'prior', 'most_frequent', 'uniform', 'constant'}
        ) or (
            not classification and type(initializer) is DummyRegressor
            and initializer.strategy in {'mean', 'median', 'quantile', 'constant'}
        )
        if not zero_initializer and not constant_initializer:
            raise ValueError('nonconstant or custom init estimators are unsupported; '
                             'use zero or a constant dummy strategy')
        raw = model._raw_predict_init(np.zeros((1, model.n_features_in_), dtype=np.float32))[0]
        result['base_scores'] = ([0.0, float(raw[0])] if classification and len(model.classes_) == 2
                                 else raw.tolist())
        result['learning_rate'] = float(model.learning_rate)
        result['sigmoid'] = 2.0 if classification and model.loss == 'exponential' else 1.0
        estimators = model.estimators_.ravel()
    else:
        estimators = model.estimators_
    trees = []
    for index, estimator in enumerate(estimators):
        tree = estimator.tree_
        exported = {'children_left': tree.children_left.tolist(),
                    'children_right': tree.children_right.tolist(),
                    'feature': tree.feature.tolist(),
                    'threshold': [export_threshold(value) for value in tree.threshold],
                    'value': tree.value.reshape(tree.node_count, -1).tolist()}
        if hasattr(tree, 'missing_go_to_left'):
            exported['missing_go_to_left'] = tree.missing_go_to_left.tolist()
        if boosted:
            exported['output'] = (1 if classification and len(model.classes_) == 2
                                  else index % model.estimators_.shape[1])
        trees.append(exported)
    result['trees'] = trees
    return result


def generate_fixtures():
    # Deliberately unbalanced, non-power-of-two training data; no external dataset.
    features = np.array([[index / 4, (index * 7 % 11) - 5] for index in range(19)], dtype=np.float32)
    targets = np.array([row[0] ** 2 - 0.7 * row[1] + (index % 3) for index, row in enumerate(features)])
    binary = np.array(['dry' if index % 3 else 'wet' for index in range(len(features))])
    categories = np.array([['low', 'medium', 'high'][index % 3] for index in range(len(features))])
    queries = np.array([[-1, 0], [0, 0], [0.625, -2], [1.5, 3], [2, -4], [3.125, 1], [4.5, 0]], dtype=np.float32)
    missing_features = np.array([[0], [1], [2], [3], [np.nan], [np.nan]], dtype=np.float32)
    missing_queries = np.array([[0], [np.nan]], dtype=np.float32)
    missing_targets = np.array([0, 0, 0, 0, 1, 1])
    cases = [
        ('random-forest-regression', RandomForestRegressor(n_estimators=3, max_depth=4, random_state=37), targets),
        ('random-forest-classification', RandomForestClassifier(n_estimators=3, max_depth=3, random_state=41), categories),
        ('gradient-boosting-regression', GradientBoostingRegressor(n_estimators=3, max_depth=2, learning_rate=0.2, random_state=43), targets),
        ('gradient-boosting-binary', GradientBoostingClassifier(n_estimators=3, max_depth=2, learning_rate=0.3, random_state=47), binary),
        ('gradient-boosting-multiclass', GradientBoostingClassifier(n_estimators=2, max_depth=2, learning_rate=0.2, random_state=53), categories),
        ('gradient-boosting-exponential', GradientBoostingClassifier(n_estimators=2, max_depth=2, loss='exponential', random_state=59), binary),
        ('random-forest-regression-missing-only', RandomForestRegressor(n_estimators=1, bootstrap=False, random_state=0), missing_targets),
        ('random-forest-classification-missing-only', RandomForestClassifier(n_estimators=1, bootstrap=False, random_state=0), missing_targets),
    ]
    fixtures = []
    for name, model, response in cases:
        missing_only = name.endswith('-missing-only')
        model.fit(missing_features if missing_only else features, response)
        test_features = missing_queries if missing_only else queries
        if not missing_only and isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
            test_features = np.vstack([queries, [np.nan, 1], [1, np.nan]]).astype(np.float32)
        classification = hasattr(model, 'classes_')
        predictions = model.predict(test_features)
        indices = ([int(np.flatnonzero(model.classes_ == label)[0]) for label in predictions]
                   if classification else predictions.tolist())
        leaves = model.apply(test_features).reshape(len(test_features), -1).astype(int).tolist()
        scores = model.predict_proba(test_features).tolist() if classification else [[value] for value in indices]
        fixtures.append({'name': name, 'model': export_model(model),
                         'features': [[None if np.isnan(value) else float(value) for value in row] for row in test_features],
                         'predictions': indices, 'scores': scores, 'leafIndices': leaves})
    destination = Path(__file__).with_name('sklearn-fixtures.json')
    destination.write_text(json.dumps({'oracle': f'scikit-learn {sklearn.__version__}', 'cases': fixtures}, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(fixtures)} independent sklearn cases to {destination.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', action='store_true')
    parser.add_argument('input', nargs='?')
    parser.add_argument('output', nargs='?')
    arguments = parser.parse_args()
    if arguments.fixtures:
        generate_fixtures()
    elif arguments.input and arguments.output:
        import joblib
        Path(arguments.output).write_text(json.dumps(export_model(joblib.load(arguments.input)), allow_nan=False, indent=2) + '\n')
    else:
        parser.error('provide --fixtures or a trusted joblib input and JSON output')
