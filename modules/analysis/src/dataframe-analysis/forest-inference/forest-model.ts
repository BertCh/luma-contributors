// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_FOREST_INFERENCE_MAXIMUM_DEPTH,
  GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT,
  GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT,
  GPU_FOREST_INFERENCE_NODE_STRIDE,
  GPU_FOREST_INFERENCE_TREE_STRIDE,
  getGPUForestInferenceParameterValues,
  validateForestInferenceModelShape,
  type GPUForestInferenceModelShape
} from './forest-inference-parameters';

const ID = 'GPUForestInference';
const MISSING = 0xffffffff;

/** Caller-owned sparse forest buffers. Upload these to graph views; no training rows are read. */
export type GPUForestInferenceModel = GPUForestInferenceModelShape & {
  nodes: Uint32Array;
  trees: Uint32Array;
  leafValues: Float32Array;
  parameters: Float32Array;
  /** Classification predictions are indices into these labels, which remain on the CPU. */
  classes?: readonly (number | string)[];
};

type ForestNode = {
  feature: number;
  left: number;
  right: number;
  threshold: number;
  flags: number;
  leafId: number;
  values: number[];
};
type ForestTree = {nodes: ForestNode[]; weight: number};
type RecordValue = Record<string, unknown>;

function requireObject(value: unknown, name: string): RecordValue {
  validateRequirement(
    ID,
    Boolean(value) && typeof value === 'object' && !Array.isArray(value),
    `${name} must be an object`
  );
  return value as RecordValue;
}
function requireArray(value: unknown, name: string): unknown[] {
  validateRequirement(ID, Array.isArray(value), `${name} must be an array`);
  return value;
}
function requireNumber(value: unknown, name: string): number {
  validateRequirement(
    ID,
    typeof value === 'number' && Number.isFinite(value),
    `${name} must be a finite number`
  );
  return value;
}
function numberArray(value: unknown, name: string): number[] {
  return requireArray(value, name).map((entry, index) => requireNumber(entry, `${name}[${index}]`));
}
/** sklearn missing-versus-present splits use infinity; JSON carries it as an explicit token. */
function readScikitLearnThresholds(value: unknown): number[] {
  return requireArray(value, 'threshold').map((threshold, index) => {
    if (threshold === 'Infinity') return Infinity;
    if (threshold === '-Infinity') return -Infinity;
    validateRequirement(
      ID,
      typeof threshold === 'number' && !Number.isNaN(threshold),
      `threshold[${index}] must be a non-NaN number or the string Infinity or -Infinity`
    );
    return threshold;
  });
}
function readLabels(value: unknown): (number | string)[] {
  const classes = requireArray(value, 'classes');
  validateRequirement(
    ID,
    classes.every(
      label => typeof label === 'string' || (typeof label === 'number' && Number.isFinite(label))
    ) && new Set(classes).size === classes.length,
    'classes must contain unique finite numbers or strings'
  );
  return classes as (number | string)[];
}
function floatBits(value: number): number {
  return new Uint32Array(Float32Array.of(value).buffer)[0];
}
/** Largest float32 <= a double threshold preserves <= routing of float32 features. */
function floorThreshold(value: number): number {
  const rounded = Math.fround(value);
  if (rounded <= value) return rounded;
  const bits = floatBits(rounded);
  return new Float32Array(
    Uint32Array.of(rounded === 0 ? 0x80000001 : rounded > 0 ? bits - 1 : bits + 1).buffer
  )[0];
}
function leafNode(leafId: number, values: number[]): ForestNode {
  return {feature: MISSING, left: MISSING, right: MISSING, threshold: 0, flags: 0, leafId, values};
}

function packModel(
  shape: Pick<GPUForestInferenceModelShape, 'featureCount' | 'outputCount' | 'task' | 'transform'>,
  forest: ForestTree[],
  baseScores: number[],
  classes?: readonly (number | string)[],
  sigmoid = 1
): GPUForestInferenceModel {
  validateCapacity(ID, 'treeCount', forest.length, 1, GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT);
  const nodeCount = forest.reduce((total, tree) => total + tree.nodes.length, 0);
  validateCapacity(ID, 'nodeCount', nodeCount, 1, GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT);
  const modelShape = {...shape, treeCount: forest.length, nodeCount, maximumDepth: 0};
  validateForestInferenceModelShape(ID, modelShape);
  validateRequirement(
    ID,
    baseScores.length === shape.outputCount,
    'baseScores length must equal outputCount'
  );
  const nodes = new Uint32Array(nodeCount * GPU_FOREST_INFERENCE_NODE_STRIDE);
  const trees = new Uint32Array(forest.length * GPU_FOREST_INFERENCE_TREE_STRIDE);
  const leafValues = new Float32Array(nodeCount * shape.outputCount);
  let nodeOffset = 0;
  forest.forEach((tree, treeIndex) => {
    validateFiniteNumber(ID, 'tree weight', tree.weight);
    const visited = new Set<number>();
    const visit = (nodeIndex: number, depth: number): void => {
      validateCapacity(ID, 'tree depth', depth, 0, GPU_FOREST_INFERENCE_MAXIMUM_DEPTH);
      validateCapacity(ID, 'child node', nodeIndex, 0, tree.nodes.length - 1);
      validateRequirement(
        ID,
        !visited.has(nodeIndex),
        'tree nodes must have one parent and no cycles'
      );
      visited.add(nodeIndex);
      modelShape.maximumDepth = Math.max(modelShape.maximumDepth, depth);
      const node = tree.nodes[nodeIndex];
      const packedOffset = (nodeOffset + nodeIndex) * GPU_FOREST_INFERENCE_NODE_STRIDE;
      if (node.feature === MISSING) {
        validateCapacity(ID, 'leafId', node.leafId, 0, MISSING - 1);
        validateRequirement(
          ID,
          node.values.length === shape.outputCount,
          'leaf values length must equal outputCount'
        );
        node.values.forEach((value, channel) => {
          validateFiniteNumber(ID, 'leaf value', value);
          leafValues[(nodeOffset + nodeIndex) * shape.outputCount + channel] = value;
        });
      } else {
        validateCapacity(ID, 'split feature', node.feature, 0, shape.featureCount - 1);
        validateRequirement(ID, !Number.isNaN(node.threshold), 'split threshold must not be NaN');
        visit(node.left, depth + 1);
        visit(node.right, depth + 1);
      }
      nodes.set(
        [
          node.feature,
          node.feature === MISSING ? MISSING : node.left + nodeOffset,
          node.feature === MISSING ? MISSING : node.right + nodeOffset,
          floatBits(node.threshold),
          node.flags,
          node.leafId
        ],
        packedOffset
      );
    };
    visit(0, 0);
    validateRequirement(
      ID,
      visited.size === tree.nodes.length,
      'tree nodes must all be reachable from root 0'
    );
    trees.set(
      [nodeOffset, nodeOffset + tree.nodes.length, floatBits(tree.weight), 0],
      treeIndex * GPU_FOREST_INFERENCE_TREE_STRIDE
    );
    nodeOffset += tree.nodes.length;
  });
  return {
    ...modelShape,
    nodes,
    trees,
    leafValues,
    parameters: getGPUForestInferenceParameterValues({baseScores, sigmoid}),
    ...(classes ? {classes} : {})
  };
}

/**
 * Reads the version-1 sklearn forest JSON emitted by test/gpu-dataframe/forest-inference/export_sklearn_forest.py.
 * Supports single-target RandomForestRegressor/Classifier and GradientBoostingRegressor/Classifier
 * (log_loss or exponential). Custom nonconstant initial estimators and multioutput targets are rejected
 * by the exporter. Split thresholds are floored to float32, preserving float32-feature routing.
 */
export function readScikitLearnForestModel(json: unknown): GPUForestInferenceModel {
  const model = requireObject(json, 'scikit-learn model');
  validateRequirement(
    ID,
    model['format'] === 'sklearn-forest' && model['version'] === 1,
    'scikit-learn format must be sklearn-forest version 1'
  );
  const estimator = model['estimator'];
  validateRequirement(
    ID,
    [
      'RandomForestRegressor',
      'RandomForestClassifier',
      'GradientBoostingRegressor',
      'GradientBoostingClassifier'
    ].includes(String(estimator)),
    'estimator must be a supported RandomForest or GradientBoosting class'
  );
  const classification = String(estimator).endsWith('Classifier');
  const boosted = String(estimator).startsWith('GradientBoosting');
  const classes = classification ? readLabels(model['classes']) : undefined;
  const outputCount = classes?.length ?? 1;
  const featureCount = requireNumber(model['n_features_in'], 'n_features_in');
  const transform =
    classification && boosted ? (outputCount === 2 ? 'sigmoid' : 'softmax') : 'identity';
  const rawTrees = requireArray(model['trees'], 'trees');
  validateCapacity(ID, 'treeCount', rawTrees.length, 1, GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT);
  validateForestInferenceModelShape(ID, {
    featureCount,
    outputCount,
    treeCount: rawTrees.length,
    nodeCount: 1,
    maximumDepth: 0,
    task: outputCount === 1 ? 'regression' : 'classification',
    transform: 'identity'
  });
  let totalNodes = 0;
  const forest = rawTrees.map((rawTree): ForestTree => {
    const tree = requireObject(rawTree, 'tree');
    const left = numberArray(tree['children_left'], 'children_left');
    totalNodes += left.length;
    validateMaximumCount(ID, 'nodeCount', totalNodes, GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT);
    const right = numberArray(tree['children_right'], 'children_right');
    const features = numberArray(tree['feature'], 'feature');
    const thresholds = readScikitLearnThresholds(tree['threshold']);
    const values = requireArray(tree['value'], 'value');
    const missing =
      tree['missing_go_to_left'] === undefined
        ? undefined
        : numberArray(tree['missing_go_to_left'], 'missing_go_to_left');
    validateRequirement(
      ID,
      [right, features, thresholds, values, ...(missing ? [missing] : [])].every(
        entries => entries.length === left.length
      ),
      'tree array lengths must equal children_left length'
    );
    const channel = boosted ? requireNumber(tree['output'], 'tree output') : 0;
    validateCapacity(ID, 'tree output', channel, 0, outputCount - 1);
    return {
      weight: boosted
        ? requireNumber(model['learning_rate'], 'learning_rate')
        : 1 / rawTrees.length,
      nodes: left.map((child, index) => {
        if (child === -1) {
          validateRequirement(ID, right[index] === -1, 'leaf children_right must equal -1');
          const rawValues = numberArray(values[index], 'leaf value');
          const scores = new Array<number>(outputCount).fill(0);
          if (!boosted && classification) {
            validateRequirement(
              ID,
              rawValues.length === outputCount && rawValues.every(value => value >= 0),
              'classifier leaf values must be nonnegative with outputCount entries'
            );
            const total = rawValues.reduce((sum, value) => sum + value, 0);
            validateRequirement(
              ID,
              total > 0 && Number.isFinite(total),
              'classifier leaf value sum must be positive and finite'
            );
            rawValues.forEach((value, output) => {
              scores[output] = value / total;
            });
          } else {
            validateRequirement(
              ID,
              rawValues.length === 1,
              'regression tree leaf value must contain 1 entry'
            );
            scores[channel] = rawValues[0];
          }
          return leafNode(index, scores);
        }
        validateRequirement(
          ID,
          !missing || missing[index] === 0 || missing[index] === 1,
          'missing_go_to_left must be 0 or 1'
        );
        return {
          feature: features[index],
          left: child,
          right: right[index],
          threshold: floorThreshold(thresholds[index]),
          flags: !boosted && missing ? 2 | missing[index] : 0,
          leafId: MISSING,
          values: []
        };
      })
    };
  });
  const baseScores = boosted
    ? numberArray(model['base_scores'], 'base_scores')
    : new Array<number>(outputCount).fill(0);
  return packModel(
    {featureCount, outputCount, task: classification ? 'classification' : 'regression', transform},
    forest,
    baseScores,
    classes,
    model['sigmoid'] === undefined ? 1 : requireNumber(model['sigmoid'], 'sigmoid')
  );
}

/** Reads numerical scalar-leaf gbtree save_model JSON (not get_dump JSON). DART and categorical/vector trees are rejected. */
export function readXGBoostForestModel(json: unknown): GPUForestInferenceModel {
  const learner = requireObject(requireObject(json, 'XGBoost model')['learner'], 'learner');
  const parameters = requireObject(learner['learner_model_param'], 'learner_model_param');
  const booster = requireObject(learner['gradient_booster'], 'gradient_booster');
  validateRequirement(
    ID,
    booster['name'] === 'gbtree',
    'XGBoost gradient_booster name must be gbtree (DART is unsupported)'
  );
  const objective = requireObject(learner['objective'], 'objective')['name'];
  validateRequirement(
    ID,
    ['reg:squarederror', 'binary:logistic', 'multi:softprob', 'multi:softmax'].includes(
      String(objective)
    ),
    'XGBoost objective must be reg:squarederror, binary:logistic, multi:softprob or multi:softmax'
  );
  validateRequirement(
    ID,
    parameters['num_target'] === undefined || Number(parameters['num_target']) === 1,
    'XGBoost num_target must equal 1'
  );
  const outputCount =
    objective === 'binary:logistic'
      ? 2
      : String(objective).startsWith('multi:')
        ? Number(parameters['num_class'])
        : 1;
  // A one-class multiclass objective still publishes classification probabilities.
  const classification = objective !== 'reg:squarederror';
  const transform =
    objective === 'binary:logistic' ? 'sigmoid' : classification ? 'softmax' : 'identity';
  const task = classification ? 'classification' : 'regression';
  const featureCount = Number(parameters['num_feature']);
  const treeModel = requireObject(booster['model'], 'gbtree model');
  const rawTrees = requireArray(treeModel['trees'], 'trees');
  validateCapacity(ID, 'treeCount', rawTrees.length, 1, GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT);
  validateForestInferenceModelShape(ID, {
    featureCount,
    outputCount,
    treeCount: rawTrees.length,
    nodeCount: 1,
    maximumDepth: 0,
    task,
    transform
  });
  const channels = numberArray(treeModel['tree_info'], 'tree_info');
  validateRequirement(
    ID,
    channels.length === rawTrees.length,
    'tree_info length must equal treeCount'
  );
  const rawBase = parameters['base_score'];
  let base: number[];
  if (typeof rawBase === 'string' && rawBase.trimStart().startsWith('[')) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawBase);
    } catch {
      validateRequirement(ID, false, 'XGBoost base_score must be a JSON vector of finite numbers');
    }
    base = numberArray(parsed, 'base_score');
  } else {
    base = [typeof rawBase === 'string' ? Number(rawBase) : requireNumber(rawBase, 'base_score')];
  }
  const rawScores = base.length === 1 ? new Array<number>(outputCount).fill(base[0]) : base;
  let baseScores = rawScores;
  if (objective === 'binary:logistic') {
    validateRequirement(
      ID,
      base.length === 1 && base[0] > 0 && base[0] < 1,
      'binary base_score must be one probability in (0, 1)'
    );
    baseScores = [0, Math.log(base[0] / (1 - base[0]))];
  }
  let totalNodes = 0;
  const forest = rawTrees.map((rawTree, treeIndex): ForestTree => {
    const tree = requireObject(rawTree, 'tree');
    const treeParameters = requireObject(tree['tree_param'], 'tree_param');
    validateRequirement(
      ID,
      [0, 1].includes(Number(treeParameters['size_leaf_vector'] ?? 1)) &&
        Number(treeParameters['num_deleted'] ?? 0) === 0,
      'XGBoost size_leaf_vector must be 0 or 1 and num_deleted must equal 0'
    );
    const left = numberArray(tree['left_children'], 'left_children');
    totalNodes += left.length;
    validateMaximumCount(ID, 'nodeCount', totalNodes, GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT);
    const right = numberArray(tree['right_children'], 'right_children');
    const features = numberArray(tree['split_indices'], 'split_indices');
    const conditions = numberArray(tree['split_conditions'], 'split_conditions');
    const defaultLeft = requireArray(tree['default_left'], 'default_left');
    const splitTypes = numberArray(tree['split_type'], 'split_type');
    validateRequirement(
      ID,
      [right, features, conditions, defaultLeft, splitTypes].every(
        values => values.length === left.length
      ) && Number(treeParameters['num_nodes']) === left.length,
      'XGBoost node array lengths must equal num_nodes'
    );
    validateRequirement(
      ID,
      splitTypes.every(value => value === 0),
      'XGBoost split_type must be numerical (0); categorical splits are unsupported'
    );
    const channel = objective === 'binary:logistic' ? 1 : channels[treeIndex];
    validateCapacity(
      ID,
      'tree_info channel',
      channels[treeIndex],
      0,
      objective === 'binary:logistic' ? 0 : outputCount - 1
    );
    return {
      weight: 1,
      nodes: left.map((child, index) => {
        if (child === -1) {
          validateRequirement(ID, right[index] === -1, 'XGBoost leaf right child must equal -1');
          const scores = new Array<number>(outputCount).fill(0);
          scores[channel] = conditions[index];
          return leafNode(index, scores);
        }
        validateRequirement(
          ID,
          [0, 1, false, true].includes(defaultLeft[index] as number | boolean),
          'XGBoost default_left must be boolean or 0/1'
        );
        return {
          feature: features[index],
          left: child,
          right: right[index],
          threshold: Math.fround(conditions[index]),
          flags: 2 | 16 | Number(Boolean(defaultLeft[index])),
          leafId: MISSING,
          values: []
        };
      })
    };
  });
  return packModel(
    {
      featureCount,
      outputCount,
      task,
      transform
    },
    forest,
    baseScores,
    classification ? Array.from({length: outputCount}, (_, index) => index) : undefined
  );
}

/** Reads numerical constant-leaf LightGBM dump_model JSON. Leaf values already include shrinkage and the initial score. */
export function readLightGBMForestModel(json: unknown): GPUForestInferenceModel {
  const model = requireObject(json, 'LightGBM model');
  const objective = String(model['objective']);
  validateRequirement(
    ID,
    !/(?:^| )sqrt(?: |$)/.test(objective),
    'LightGBM regression sqrt link is unsupported'
  );
  const binary = /^binary(?: |$)/.test(objective);
  const multiclass = /^multiclass(?: |$)/.test(objective);
  validateRequirement(
    ID,
    binary ||
      multiclass ||
      /^(regression|regression_l1|huber|fair|quantile|mape)(?: |$)/.test(objective),
    'LightGBM objective must be regression with identity link, binary or multiclass'
  );
  validateRequirement(
    ID,
    !model['average_output'],
    'LightGBM average_output must be false (random-forest boosting is unsupported)'
  );
  const outputCount = binary ? 2 : multiclass ? Number(model['num_class']) : 1;
  const perIteration = binary ? 1 : outputCount;
  validateRequirement(
    ID,
    Number(model['num_tree_per_iteration']) === perIteration,
    'LightGBM num_tree_per_iteration must equal raw output count'
  );
  const featureCount = Number(model['max_feature_idx']) + 1;
  const rawTrees = requireArray(model['tree_info'], 'tree_info');
  validateCapacity(ID, 'treeCount', rawTrees.length, 1, GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT);
  validateRequirement(
    ID,
    rawTrees.length % perIteration === 0,
    'LightGBM treeCount must contain complete multiclass iterations'
  );
  validateForestInferenceModelShape(ID, {
    featureCount,
    outputCount,
    treeCount: rawTrees.length,
    nodeCount: 1,
    maximumDepth: 0,
    task: outputCount === 1 ? 'regression' : 'classification',
    transform: 'identity'
  });
  let totalNodes = 0;
  const forest = rawTrees.map((rawTree, treeIndex): ForestTree => {
    const tree = requireObject(rawTree, 'tree');
    validateRequirement(ID, !tree['is_linear'], 'LightGBM is_linear must be false');
    const channel = binary ? 1 : treeIndex % perIteration;
    const nodes: ForestNode[] = [];
    const append = (rawNode: unknown, depth: number): number => {
      validateCapacity(ID, 'tree depth', depth, 0, GPU_FOREST_INFERENCE_MAXIMUM_DEPTH);
      totalNodes++;
      validateMaximumCount(ID, 'nodeCount', totalNodes, GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT);
      const node = requireObject(rawNode, 'tree node');
      const index = nodes.length;
      if (node['leaf_value'] !== undefined) {
        validateRequirement(
          ID,
          node['leaf_coeff'] === undefined && node['leaf_const'] === undefined,
          'LightGBM linear leaves are unsupported'
        );
        const scores = new Array<number>(outputCount).fill(0);
        scores[channel] = requireNumber(node['leaf_value'], 'leaf_value');
        nodes.push(
          leafNode(
            node['leaf_index'] === undefined && depth === 0
              ? 0
              : requireNumber(node['leaf_index'], 'leaf_index'),
            scores
          )
        );
      } else {
        validateRequirement(
          ID,
          node['decision_type'] === '<=',
          'LightGBM decision_type must be <=; categorical splits are unsupported'
        );
        validateRequirement(
          ID,
          ['None', 'Zero', 'NaN'].includes(String(node['missing_type'])) &&
            typeof node['default_left'] === 'boolean',
          'LightGBM missing_type must be None, Zero or NaN and default_left must be boolean'
        );
        nodes.push({
          feature: requireNumber(node['split_feature'], 'split_feature'),
          left: 0,
          right: 0,
          threshold: floorThreshold(requireNumber(node['threshold'], 'threshold')),
          flags:
            2 |
            Number(node['default_left']) |
            (node['missing_type'] === 'Zero' ? 4 : 0) |
            (node['missing_type'] !== 'NaN' ? 8 : 0),
          leafId: MISSING,
          values: []
        });
        nodes[index].left = append(node['left_child'], depth + 1);
        nodes[index].right = append(node['right_child'], depth + 1);
      }
      return index;
    };
    append(tree['tree_structure'], 0);
    return {nodes, weight: 1};
  });
  const sigmoidMatch = /(?:^| )sigmoid:([^ ]+)/.exec(objective);
  const sigmoid = sigmoidMatch ? Number(sigmoidMatch[1]) : 1;
  return packModel(
    {
      featureCount,
      outputCount,
      task: outputCount > 1 ? 'classification' : 'regression',
      transform: binary ? 'sigmoid' : multiclass ? 'softmax' : 'identity'
    },
    forest,
    new Array<number>(outputCount).fill(0),
    outputCount > 1 ? Array.from({length: outputCount}, (_, index) => index) : undefined,
    sigmoid
  );
}
