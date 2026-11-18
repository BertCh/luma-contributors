// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/**
 * Most folds of a conformal calibration set: the interval kernel visits every fold in a fixed
 * loop, and `GPUSpatialCrossValidation` produces at most 64 folds.
 */
export const GPU_CONFORMAL_MAXIMUM_FOLD_COUNT = 64;

/** Number of float32 elements in a conformal parameter buffer. */
export const GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH = 4;

/** Per-frame settings packed for the conformal interval contributors. */
export type GPUConformalPredictionSettings = {
  /** Target miscoverage `alpha` in the open interval (0, 1): intervals cover at least `1 - alpha`. */
  miscoverage: number;
  /**
   * Localized intervals only: weight of the query location itself in the weighted quantile, the
   * kernel value at distance zero. Default 1, right for kernels scaled so that `K(0) = 1`.
   */
  selfWeight?: number;
};

/**
 * Packs conformal parameters into the 4-element float32 layout read by the conformal interval
 * contributors: `[miscoverage, selfWeight, 0, 0]`. Write it into a `GPUParameterBuffer` between
 * encodings to change the target coverage without recompiling.
 *
 * @throws If `miscoverage` is not in (0, 1) after rounding to float32, or `selfWeight` is not a
 *   non-negative finite number.
 */
export function getGPUConformalPredictionParameterValues(
  settings: GPUConformalPredictionSettings,
  target: Float32Array = new Float32Array(GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH)
): Float32Array {
  const miscoverage = Math.fround(settings.miscoverage);
  const selfWeight = Math.fround(settings.selfWeight ?? 1);
  if (!(miscoverage > 0 && miscoverage < 1)) {
    throw new Error('getGPUConformalPredictionParameterValues miscoverage must be in (0, 1)');
  }
  if (!(Number.isFinite(selfWeight) && selfWeight >= 0)) {
    throw new Error(
      'getGPUConformalPredictionParameterValues selfWeight must be a non-negative finite number'
    );
  }
  if (target.length < GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH) {
    throw new Error(
      `getGPUConformalPredictionParameterValues target needs ${GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH} elements`
    );
  }
  target.set([miscoverage, selfWeight, 0, 0]);
  return target;
}

/**
 * Calibration rows of the conformal contributors. A calibration row has a held-out prediction:
 * `predictions` is fold-major (`predictions[fold * rows + row]`), the layout of
 * `GPUSpatialCrossValidatedOLS.output.predictions`, and row `i` is scored by the model of fold
 * `folds[i]`, which did not train on it. A split (holdout) calibration set has one fold.
 */
export type GPUConformalCalibration = {
  /** Observed value per calibration row. A non-finite value excludes the row. */
  observations: GraphDataView<'float32'>;
  /** Fold-major predictions of every fold's model at every row: `foldCount * rows` values. */
  predictions: GraphDataView<'float32'>;
  /**
   * Fold of each row, for example `GPUSpatialCrossValidation.output.folds`. A fold of
   * `foldCount` or more (0xffffffff marks an unassigned row) excludes the row. Required when
   * `foldCount` is more than one; omitted means every row is in fold 0.
   */
  folds?: GraphDataView<'uint32'>;
  /**
   * Optional heteroscedastic scale per calibration row, for example the kriging standard
   * deviation. Scores become `|y - prediction| / scale`; a scale that is not a positive finite
   * number excludes the row.
   */
  scales?: GraphDataView<'float32'>;
  /** Optional row filter: nonzero keeps the row. */
  mask?: GraphDataView<'uint32'>;
};

/** Query rows of the conformal contributors. */
export type GPUConformalQuery = {
  /** Fold-major predictions of every fold's model at every query row: `foldCount * queryRows`. */
  predictions: GraphDataView<'float32'>;
  /** Optional scale per query row, matching `calibration.scales`. Default 1. */
  scales?: GraphDataView<'float32'>;
};

/** Validates calibration and query views and returns `[calibrationRows, queryRows]`. @internal */
export function validateConformalInputs(
  id: string,
  calibration: GPUConformalCalibration,
  query: GPUConformalQuery,
  foldCount: number,
  queryRows: number
): number {
  validateCapacity(id, 'foldCount', foldCount, 1, GPU_CONFORMAL_MAXIMUM_FOLD_COUNT);
  const rows = calibration.observations.length;
  validateCapacity(id, 'calibration rows', rows, 1);
  validateSinglePackedView(calibration.observations, ['float32'], `${id} calibration.observations`);
  validateSinglePackedView(calibration.predictions, ['float32'], `${id} calibration.predictions`);
  validateViewLength(id, 'calibration.predictions', calibration.predictions, foldCount * rows);
  if (foldCount > 1 && !calibration.folds) {
    throw new Error(`${id} calibration.folds is required when foldCount is more than one`);
  }
  for (const [name, view, format] of [
    ['folds', calibration.folds, 'uint32'],
    ['scales', calibration.scales, 'float32'],
    ['mask', calibration.mask, 'uint32']
  ] as const) {
    if (view) {
      validateSinglePackedView(view, [format], `${id} calibration.${name}`);
      validateViewLength(id, `calibration.${name}`, view, rows);
    }
  }
  validateSinglePackedView(query.predictions, ['float32'], `${id} query.predictions`);
  validateViewLength(id, 'query.predictions', query.predictions, foldCount * queryRows);
  if (query.scales) {
    validateSinglePackedView(query.scales, ['float32'], `${id} query.scales`);
    validateViewLength(id, 'query.scales', query.scales, queryRows);
  }
  return rows;
}

/** Returns every defined view of the calibration and query inputs. @internal */
export function getConformalInputViews(
  calibration: GPUConformalCalibration,
  query: GPUConformalQuery
): GraphDataView[] {
  const views: (GraphDataView | undefined)[] = [
    calibration.observations,
    calibration.predictions,
    calibration.folds,
    calibration.scales,
    calibration.mask,
    query.predictions,
    query.scales
  ];
  return views.filter((view): view is GraphDataView => view !== undefined);
}

/** Scratch views produced by {@link createConformalScoreNodes}. @internal */
export type ConformalScoreViews = {
  /** Fold of each calibration row, or `foldCount` for an excluded row. */
  foldKeys: GraphDataView<'uint32'>;
};

/**
 * Scores the calibration rows: `|y - prediction[fold(i)]| / scale` as sortable bit-pattern keys (score zero for excluded
 * rows), with excluded rows marked by a fold key of `foldCount`. One thread per calibration row.
 *
 * @internal
 */
export function createConformalScoreNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  calibration: GPUConformalCalibration,
  foldCount: number
): {
  nodes: GPUCommandNode<Parameters>[];
  views: ConformalScoreViews;
  scoreKeys: GraphDataView<'uint32'>;
  indices: GraphDataView<'uint32'>;
} {
  const rows = calibration.observations.length;
  const foldKeys = createTransientView<'uint32', Parameters>(
    graph,
    `${id}-fold-keys`,
    'uint32',
    rows
  );
  const scoreKeys = createTransientView<'uint32', Parameters>(
    graph,
    `${id}-score-keys`,
    'uint32',
    rows
  );
  const indices = createTransientView<'uint32', Parameters>(graph, `${id}-indices`, 'uint32', rows);
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-scores`,
    operation: 'GPUConformalPredictionIntervals',
    variant: 'scores',
    invocationCount: rows,
    bindings: [
      {name: 'observations', view: calibration.observations, type: 'f32', access: 'read'},
      {name: 'predictions', view: calibration.predictions, type: 'f32', access: 'read'},
      ...(calibration.folds
        ? [{name: 'folds', view: calibration.folds, type: 'u32' as const, access: 'read' as const}]
        : []),
      ...(calibration.scales
        ? [
            {
              name: 'scales',
              view: calibration.scales,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(calibration.mask
        ? [{name: 'mask', view: calibration.mask, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'foldKeys', view: foldKeys, type: 'u32', access: 'read_write'},
      {name: 'scoreKeys', view: scoreKeys, type: 'u32', access: 'read_write'},
      {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
    ],
    declarations: IS_FINITE_BITS_WGSL,
    body: `let fold = ${calibration.folds ? 'folds[foldsOffset + index]' : '0u'};
  var valid = fold < ${foldCount}u${calibration.mask ? ' && mask[maskOffset + index] != 0u' : ''};
  let observation = observations[observationsOffset + index];
  let prediction = predictions[predictionsOffset + min(fold, ${foldCount - 1}u) * ${rows}u + index];
  valid = valid && isFiniteBits(observation) && isFiniteBits(prediction);
  var score = abs(observation - prediction);
  ${
    calibration.scales
      ? `let scale = scales[scalesOffset + index];
  valid = valid && isFiniteBits(scale) && scale > 0.0;
  score = score / scale;`
      : ''
  }
  valid = valid && isFiniteBits(score);
  // A non-negative finite float32 sorts as its bit pattern; the score is recovered with bitcast<f32>.
  scoreKeys[scoreKeysOffset + index] = select(0u, bitcast<u32>(score), valid);
  foldKeys[foldKeysOffset + index] = select(${foldCount}u, fold, valid);
  indices[indicesOffset + index] = index;`
  });
  return {nodes: [node], views: {foldKeys}, scoreKeys, indices};
}
