// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import handwrittenFixtures from './handwritten-fixtures.json';
import {
  readXGBoostForestModel,
  readLightGBMForestModel,
  type GPUForestInferenceModel
} from '../../../src/dataframe-analysis/forest-inference/index';

/** Native one-class multiclass models still apply softmax and publish label zero. */
export function getOneClassXGBoostModelJSON(objective: 'multi:softprob' | 'multi:softmax') {
  const model = structuredClone(handwrittenFixtures.xgboostBinary);
  return {
    ...model,
    learner: {
      ...model.learner,
      learner_model_param: {
        ...model.learner.learner_model_param,
        num_class: '1',
        base_score: '[0.5]'
      },
      objective: {name: objective, softmax_multiclass_param: {num_class: '1'}}
    }
  };
}

/** Hand-derived regression and multiclass cases use native numerical JSON; no native library dependency. */
export function getNativeForestCases(): {
  name: string;
  model: GPUForestInferenceModel;
  features: number[][];
  predictions: number[];
  scores: number[][];
  leaves: number[][];
}[] {
  const xgboostRegression = structuredClone(handwrittenFixtures.xgboostBinary);
  xgboostRegression.learner.objective.name = 'reg:squarederror';
  xgboostRegression.learner.learner_model_param.base_score = '3';
  const lightgbmRegression = structuredClone(handwrittenFixtures.lightgbmBinary);
  lightgbmRegression.objective = 'regression';
  const raw = [0.75, -0.5, 1.75];
  const denominator = raw.reduce((sum, value) => sum + Math.exp(value), 0);
  const probabilities = raw.map(value => Math.exp(value) / denominator);
  const xgboostMulticlass = {
    learner: {
      learner_model_param: {
        num_feature: '1',
        num_class: '3',
        num_target: '1',
        base_score: '[0.25,0.5,-0.25]'
      },
      objective: {name: 'multi:softprob'},
      gradient_booster: {
        name: 'gbtree',
        model: {
          tree_info: [0, 1, 2],
          trees: [0.5, -1, 2].map(value => ({
            tree_param: {num_nodes: '1', num_deleted: '0', size_leaf_vector: '1'},
            left_children: [-1],
            right_children: [-1],
            split_indices: [0],
            split_conditions: [value],
            default_left: [0],
            split_type: [0]
          }))
        }
      }
    }
  };
  const lightgbmMulticlass = {
    objective: 'multiclass num_class:3',
    num_class: 3,
    num_tree_per_iteration: 3,
    max_feature_idx: 0,
    average_output: false,
    tree_info: raw.map(value => ({
      num_leaves: 1,
      shrinkage: 1,
      tree_structure: {leaf_value: value}
    }))
  };
  return [
    ...(['multi:softprob', 'multi:softmax'] as const).map(objective => ({
      name: `XGBoost one-class ${objective}`,
      model: readXGBoostForestModel(getOneClassXGBoostModelJSON(objective)),
      features: [[0], [1]],
      predictions: [0, 0],
      scores: [[1], [1]],
      leaves: [[1], [2]]
    })),
    {
      name: 'XGBoost regression scalar base score',
      model: readXGBoostForestModel(xgboostRegression),
      features: [[0], [1]],
      predictions: [1, 5],
      scores: [[1], [5]],
      leaves: [[1], [2]]
    },
    {
      name: 'LightGBM regression',
      model: readLightGBMForestModel(lightgbmRegression),
      features: [[0], [2]],
      predictions: [-1, 1],
      scores: [[-1], [1]],
      leaves: [[0], [1]]
    },
    {
      name: 'XGBoost multiclass vector base score',
      model: readXGBoostForestModel(xgboostMulticlass),
      features: [[0]],
      predictions: [2],
      scores: [probabilities],
      leaves: [[0, 0, 0]]
    },
    {
      name: 'LightGBM multiclass initial-score leaves',
      model: readLightGBMForestModel(lightgbmMulticlass),
      features: [[0]],
      predictions: [2],
      scores: [probabilities],
      leaves: [[0, 0, 0]]
    }
  ];
}
