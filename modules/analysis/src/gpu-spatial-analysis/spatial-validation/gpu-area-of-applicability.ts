// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionAllocations} from '../spatial-regression/spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';

/** Most predictors of the dissimilarity index. */
export const GPU_AREA_OF_APPLICABILITY_MAXIMUM_PREDICTOR_COUNT = 16;
/** Most training rows: the training distances are exhaustive, `O(rows^2 x predictors)`. */
export const GPU_AREA_OF_APPLICABILITY_MAXIMUM_TRAINING_ROWS = 65536;
/** Number of float32 values of {@link GPUAreaOfApplicabilityOutput.summary}. */
export const GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH = 6;

/** Status bit: fewer than two usable training rows. */
export const GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS = 1;
/** Status bit: a predictor has no variance among the usable training rows. */
export const GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR = 2;
/** Status bit: no training row has a neighbour outside its fold, so there is no threshold. */
export const GPU_AREA_OF_APPLICABILITY_STATUS_NO_THRESHOLD = 4;

const OPERATION = 'GPUAreaOfApplicability';

/** Cross-validation structure that restricts each training point's nearest neighbour. */
export type GPUAreaOfApplicabilityFolds = {
  /** Fold in which each training row is held out, `GPUSpatialCrossValidation` `output.folds`. */
  folds: GraphDataView<'uint32'>;
  /** Fold-major training masks, `GPUSpatialCrossValidation` `output.training`. */
  training: GraphDataView<'uint32'>;
  /** Number of folds. */
  foldCount: number;
};

/** Caller-owned outputs of {@link GPUAreaOfApplicability}. */
export type GPUAreaOfApplicabilityOutput = {
  /** Dissimilarity index of every training row; NaN for a row without a usable neighbour. */
  trainDissimilarity: GraphDataView<'float32'>;
  /** Dissimilarity index per query; NaN for incomplete rows or when no finite squared distance is computed. */
  queryDissimilarity: GraphDataView<'float32'>;
  /** 1 where the query row is inside the area of applicability (index at most the threshold). */
  applicable: GraphDataView<'uint32'>;
  /**
   * `[meanDistance, threshold, usableTrainingRows, scoredTrainingRows, firstQuartile,
   * thirdQuartile]`: the mean over training rows of the mean distance to all other training rows,
   * the threshold, and the quartiles of the training index (type 7, as R `quantile`).
   */
  summary: GraphDataView<'float32'>;
  /** One status word, see the `GPU_AREA_OF_APPLICABILITY_STATUS_*` constants. */
  status: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUAreaOfApplicability}. */
export type GPUAreaOfApplicabilityProps = {
  id?: string;
  /** Training predictors, `rows x predictorCount` row-major. A row with a non-finite value is unusable. */
  training: GraphDataView<'float32'>;
  /** Query predictors, `queryRows x predictorCount` row-major. */
  query: GraphDataView<'float32'>;
  predictorCount: number;
  /**
   * Optional variable importance per predictor (non-negative), multiplying the standardized
   * predictor before distances are taken. Default: all 1.
   */
  weights?: GraphDataView<'float32'>;
  /** Optional training row filter: nonzero keeps the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Cross-validation folds. A training point's nearest neighbour is then searched among the
   * training rows of the fold in which it is held out; a row held out in no fold is not scored.
   * Without folds every other training row is a candidate.
   */
  crossValidation?: GPUAreaOfApplicabilityFolds;
  output: GPUAreaOfApplicabilityOutput;
};

/**
 * Dissimilarity index and area of applicability (Meyer and Pebesma 2021, "Predicting into
 * unknown space? Estimating the area of applicability of spatial prediction models", Methods in
 * Ecology and Evolution 12(9); definition of the R package CAST 1.2.0).
 *
 * Predictors are standardized with the sample mean and standard deviation (`n - 1`) of the
 * usable training rows and multiplied by the variable weights. The index of a point is its
 * Euclidean distance to the nearest training point, divided by `meanDistance`, the mean over
 * training rows of their mean distance to all other training rows. For a training point the
 * nearest neighbour is searched among the training rows of the cross-validation fold in which it
 * is held out (the fold's training mask, so a spatial buffer is honored), else among all other
 * training rows. The threshold is `min(Q3 + 1.5 IQR, max)` of the training indices with
 * quartiles of type 7; a query row is inside the area when its index is at most the threshold.
 *
 * Differences from CAST: Euclidean distance only (no Mahalanobis or Gower), categorical
 * predictors must be supplied as dummy columns, a training row held out in no fold or without
 * a neighbour is left out of the threshold instead of receiving an infinite index, and rows with
 * non-finite predictors are unusable rather than imputed. Distances are exhaustive: the training
 * pass costs `O(rows^2 x predictors)` and each query row `O(rows x predictors)`; arithmetic is
 * float32. This route accepts at most 65,536 training rows, 16 predictors, and, when
 * crossValidation is supplied, 64 folds.
 */
export class GPUAreaOfApplicability implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rows: number;
  readonly queryRows: number;
  readonly props: GPUAreaOfApplicabilityProps;
  constructor(props: GPUAreaOfApplicabilityProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'training',
      'query',
      'predictorCount',
      'weights',
      'mask',
      'crossValidation',
      'output'
    ]);
    props = Object.freeze({
      ...captured,
      output: snapshotConstructionFields(captured.output, [
        'trainDissimilarity',
        'queryDissimilarity',
        'applicable',
        'summary',
        'status'
      ]),
      ...(captured.crossValidation
        ? {
            crossValidation: snapshotConstructionFields(captured.crossValidation, [
              'folds',
              'training',
              'foldCount'
            ])
          }
        : {})
    });
    this.props = props;
    this.id = props.id ?? 'area-of-applicability';
    const {id} = this;
    const columns = props.predictorCount;
    validateCapacity(
      id,
      'predictorCount',
      columns,
      1,
      GPU_AREA_OF_APPLICABILITY_MAXIMUM_PREDICTOR_COUNT
    );
    validateSinglePackedView(props.training, ['float32'], `${id} training`);
    validateSinglePackedView(props.query, ['float32'], `${id} query`);
    this.rows = props.training.length / columns;
    this.queryRows = props.query.length / columns;
    validateCapacity(
      id,
      'training rows',
      this.rows,
      2,
      GPU_AREA_OF_APPLICABILITY_MAXIMUM_TRAINING_ROWS
    );
    validateCapacity(id, 'query rows', this.queryRows, 1);
    if (props.weights) {
      validateSinglePackedView(props.weights, ['float32'], `${id} weights`);
      validateViewLength(id, 'weights', props.weights, columns);
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', props.mask, this.rows);
    }
    const cross = props.crossValidation;
    if (cross) {
      validateCapacity(id, 'crossValidation.foldCount', cross.foldCount, 1, 64);
      validateSinglePackedView(cross.folds, ['uint32'], `${id} crossValidation.folds`);
      validateViewLength(id, 'crossValidation.folds', cross.folds, this.rows);
      validateSinglePackedView(cross.training, ['uint32'], `${id} crossValidation.training`);
      validateViewLength(
        id,
        'crossValidation.training',
        cross.training,
        cross.foldCount * this.rows
      );
    }
    const {output} = props;
    for (const [name, view, format, length] of [
      ['trainDissimilarity', output.trainDissimilarity, 'float32', this.rows],
      ['queryDissimilarity', output.queryDissimilarity, 'float32', this.queryRows],
      ['applicable', output.applicable, 'uint32', this.queryRows],
      ['summary', output.summary, 'float32', GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH],
      ['status', output.status, 'uint32', 1]
    ] as const) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
    validateRegressionOutputDispatches(this.id, output, [
      ['summary', 'status'],
      {read: ['summary'], write: ['trainDissimilarity']},
      {read: ['summary'], write: ['queryDissimilarity', 'applicable']}
    ]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(output), this.getInputViews());
  }

  private getInputViews(): GraphDataView[] {
    const {props} = this;
    const views: (GraphDataView | undefined)[] = [
      props.training,
      props.query,
      props.weights,
      props.mask,
      props.crossValidation?.folds,
      props.crossValidation?.training
    ];
    return views.filter((view): view is GraphDataView => view !== undefined);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows, queryRows} = this;
    const columns = props.predictorCount;
    const cross = props.crossValidation;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(this.getInputViews().map((view, index) => [`input ${index}`, view])),
      ...output
    });
    validateRegressionAllocations(graph.device, id, {
      'training row scratch, keys and published sort/scan scratch': rows,
      'feature moments': 2 * columns,
      'scaled training': rows * columns,
      'scaled query': queryRows * columns
    });
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...Object.values(output)]);
    const transient = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView<Format, Parameters>(graph, `${id}-${name}`, format, length);
    const rowValid = transient('row-valid', 'uint32', rows);
    const moments = transient('moments', 'float32', 2 * columns);
    const scaled = transient('scaled', 'float32', rows * columns);
    const scaledQuery = transient('scaled-query', 'float32', queryRows * columns);
    const nearest = transient('nearest', 'float32', rows);
    const mean = transient('mean', 'float32', rows);
    const keys = transient('keys', 'uint32', rows);
    const indices = transient('indices', 'uint32', rows);
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-sort`,
      operation: OPERATION,
      indices,
      keys: [{view: keys, keyBits: 32}]
    });
    const weightBinding = props.weights
      ? [{name: 'weights', view: props.weights, type: 'f32' as const, access: 'read' as const}]
      : [];
    const weightFactor = (column: string) =>
      props.weights ? `weights[weightsOffset + ${column}]` : '1.0';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-valid`,
        operation: OPERATION,
        variant: 'usable-rows',
        invocationCount: rows,
        bindings: [
          {name: 'training', view: props.training, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'rowValid', view: rowValid, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `var valid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  for (var column = 0u; column < ${columns}u; column++) {
    valid = valid && isFiniteBits(training[trainingOffset + index * ${columns}u + column]);
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, valid);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-moments`,
        operation: OPERATION,
        variant: 'moments',
        invocationCount: columns,
        bindings: [
          {name: 'training', view: props.training, type: 'f32', access: 'read'},
          {name: 'rowValid', view: rowValid, type: 'u32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
        ],
        declarations: ROUNDED_DIFFERENCE_WGSL,
        // Center both passes around a usable value; identical packed values then have exact SD=0.
        body: `var count = 0.0;
  var sum = 0.0;
  var origin = 0.0;
  let runtimeZero = rowValid[rowValidOffset] >> 1u;
  for (var row = 0u; row < ${rows}u; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let value = training[trainingOffset + row * ${columns}u + index];
      if (count == 0.0) { origin = value; }
      count += 1.0;
      sum += getRoundedDifference(value, origin, runtimeZero);
    }
  }
  let centeredMean = sum / count;
  var squares = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let centered = getRoundedDifference(training[trainingOffset + row * ${columns}u + index], origin, runtimeZero);
      let deviation = getRoundedDifference(centered, centeredMean, runtimeZero);
      squares += deviation * deviation;
    }
  }
  moments[momentsOffset + index * 2u] = origin + centeredMean;
  moments[momentsOffset + index * 2u + 1u] = select(0.0, sqrt(squares / (count - 1.0)), count > 1.0);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scale`,
        operation: OPERATION,
        variant: 'standardize-training',
        invocationCount: rows * columns,
        bindings: [
          {name: 'training', view: props.training, type: 'f32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          ...weightBinding,
          {name: 'scaled', view: scaled, type: 'f32', access: 'read_write'}
        ],
        body: `let column = index % ${columns}u;
  let deviation = training[trainingOffset + index] - moments[momentsOffset + column * 2u];
  let standardized = deviation / moments[momentsOffset + column * 2u + 1u];
  scaled[scaledOffset + index] = ${weightFactor('column')} * standardized;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scale-query`,
        operation: OPERATION,
        variant: 'standardize-query',
        invocationCount: queryRows * columns,
        bindings: [
          {name: 'query', view: props.query, type: 'f32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          ...weightBinding,
          {name: 'scaledQuery', view: scaledQuery, type: 'f32', access: 'read_write'}
        ],
        body: `let column = index % ${columns}u;
  let deviation = query[queryOffset + index] - moments[momentsOffset + column * 2u];
  let standardized = deviation / moments[momentsOffset + column * 2u + 1u];
  scaledQuery[scaledQueryOffset + index] = ${weightFactor('column')} * standardized;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-training-distances`,
        operation: OPERATION,
        variant: 'training-distances',
        invocationCount: rows,
        bindings: [
          {name: 'scaled', view: scaled, type: 'f32', access: 'read'},
          {name: 'rowValid', view: rowValid, type: 'u32', access: 'read'},
          ...(cross
            ? [
                {
                  name: 'testFolds',
                  view: cross.folds,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'trainingMasks',
                  view: cross.training,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'nearest', view: nearest, type: 'f32', access: 'read_write'},
          {name: 'mean', view: mean, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        // `mean` is over every other usable row; `nearest` over the candidate rows only.
        body: `var nearestSquared = 3.4e38;
  var found = false;
  var total = 0.0;
  var others = 0.0;
  let usable = rowValid[rowValidOffset + index] != 0u;
  ${
    cross
      ? `let testFold = testFolds[testFoldsOffset + index];
  let scored = testFold < ${cross.foldCount}u;`
      : 'let scored = true;'
  }
  if (usable) {
    for (var row = 0u; row < ${rows}u; row++) {
      if (row == index || rowValid[rowValidOffset + row] == 0u) { continue; }
      var squared = 0.0;
      for (var column = 0u; column < ${columns}u; column++) {
        let difference = scaled[scaledOffset + index * ${columns}u + column] - scaled[scaledOffset + row * ${columns}u + column];
        squared += difference * difference;
      }
      total += sqrt(squared);
      others += 1.0;
      ${
        cross
          ? `if (scored && trainingMasks[trainingMasksOffset + min(testFold, ${cross.foldCount - 1}u) * ${rows}u + row] != 0u && isFiniteBits(squared) && (!found || squared < nearestSquared)) {`
          : 'if (isFiniteBits(squared) && (!found || squared < nearestSquared)) {'
      }
        nearestSquared = squared;
        found = true;
      }
    }
  }
  nearest[nearestOffset + index] = select(getNaN(), sqrt(nearestSquared), usable && scored && found);
  mean[meanOffset + index] = select(getNaN(), total / others, usable && others > 0.0);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean-distance`,
        operation: OPERATION,
        variant: 'mean-distance',
        invocationCount: 1,
        bindings: [
          {name: 'mean', view: mean, type: 'f32', access: 'read'},
          {name: 'rowValid', view: rowValid, type: 'u32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
        body: `var total = 0.0;
  var count = 0.0;
  var usable = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    usable += f32(rowValid[rowValidOffset + row]);
    let value = mean[meanOffset + row];
    if (isFiniteBits(value)) {
      total += value;
      count += 1.0;
    }
  }
  var code = 0u;
  if (usable < 2.0) { code |= ${GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS}u; }
  for (var column = 0u; column < ${columns}u; column++) {
    if (!(moments[momentsOffset + column * 2u + 1u] > 0.0)) { code |= ${GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR}u; }
  }
  let meanDistance = select(getNaN(), total / count, count > 0.0 && code == 0u);
  for (var element = 0u; element < ${GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH}u; element++) {
    summary[summaryOffset + element] = getNaN();
  }
  summary[summaryOffset] = meanDistance;
  summary[summaryOffset + 2u] = usable;
  status[statusOffset] = code;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-training-index`,
        operation: OPERATION,
        variant: 'training-index',
        invocationCount: rows,
        bindings: [
          {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read'},
          {
            name: 'trainDissimilarity',
            view: output.trainDissimilarity,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
        // Indices are non-negative, so their bit patterns sort as numbers; 0xffffffff (a NaN) sorts last.
        body: `let meanDistance = summary[summaryOffset];
  let value = nearest[nearestOffset + index] / meanDistance;
  let valid = isFiniteBits(value) && meanDistance > 0.0;
  trainDissimilarity[trainDissimilarityOffset + index] = select(getNaN(), value, valid);
  keys[keysOffset + index] = select(0xffffffffu, bitcast<u32>(value), valid);
  indices[indicesOffset + index] = index;`
      }),
      ...sorted.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-threshold`,
        operation: OPERATION,
        variant: 'threshold',
        invocationCount: 1,
        bindings: [
          {name: 'sortedKeys', view: sorted.sortedPrimaryKeys, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        // R quantile type 7 at probability p: x[floor(h)] + (h - floor(h)) (x[floor(h) + 1] - x[floor(h)]), h = (m - 1) p.
        body: `var scoredCount = 0u;
  for (var row = 0u; row < ${rows}u; row++) {
    if (sortedKeys[sortedKeysOffset + row] != 0xffffffffu) { scoredCount++; }
  }
  summary[summaryOffset + 3u] = f32(scoredCount);
  if (scoredCount == 0u) {
    status[statusOffset] = status[statusOffset] | ${GPU_AREA_OF_APPLICABILITY_STATUS_NO_THRESHOLD}u;
  } else {
    let span = f32(scoredCount - 1u);
    var quartiles = vec2f(0.0, 0.0);
    for (var which = 0u; which < 2u; which++) {
      let position = span * select(0.25, 0.75, which == 1u);
      let lowerIndex = u32(floor(position));
      let upperIndex = min(lowerIndex + 1u, scoredCount - 1u);
      let lowerValue = bitcast<f32>(sortedKeys[sortedKeysOffset + lowerIndex]);
      let upperValue = bitcast<f32>(sortedKeys[sortedKeysOffset + upperIndex]);
      quartiles[which] = lowerValue + (position - floor(position)) * (upperValue - lowerValue);
    }
    let maximum = bitcast<f32>(sortedKeys[sortedKeysOffset + scoredCount - 1u]);
    summary[summaryOffset + 1u] = min(quartiles.y + 1.5 * (quartiles.y - quartiles.x), maximum);
    summary[summaryOffset + 4u] = quartiles.x;
    summary[summaryOffset + 5u] = quartiles.y;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-query`,
        operation: OPERATION,
        variant: 'query-index',
        invocationCount: queryRows,
        bindings: [
          {name: 'query', view: props.query, type: 'f32', access: 'read'},
          {name: 'scaled', view: scaled, type: 'f32', access: 'read'},
          {name: 'scaledQuery', view: scaledQuery, type: 'f32', access: 'read'},
          {name: 'rowValid', view: rowValid, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read'},
          {
            name: 'queryDissimilarity',
            view: output.queryDissimilarity,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'applicable', view: output.applicable, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
        body: `var complete = true;
  for (var column = 0u; column < ${columns}u; column++) {
    complete = complete && isFiniteBits(query[queryOffset + index * ${columns}u + column]);
  }
  var nearestSquared = 3.4e38;
  var found = false;
  if (complete) {
    for (var row = 0u; row < ${rows}u; row++) {
      if (rowValid[rowValidOffset + row] == 0u) { continue; }
      var squared = 0.0;
      for (var column = 0u; column < ${columns}u; column++) {
        let difference = scaledQuery[scaledQueryOffset + index * ${columns}u + column] - scaled[scaledOffset + row * ${columns}u + column];
        squared += difference * difference;
      }
      if (isFiniteBits(squared) && (!found || squared < nearestSquared)) {
        nearestSquared = squared;
        found = true;
      }
    }
  }
  let meanDistance = summary[summaryOffset];
  let value = sqrt(nearestSquared) / meanDistance;
  let valid = complete && found && isFiniteBits(value) && meanDistance > 0.0;
  queryDissimilarity[queryDissimilarityOffset + index] = select(getNaN(), value, valid);
  applicable[applicableOffset + index] = select(0u, 1u, valid && value <= summary[summaryOffset + 1u]);`
      })
    ];
  }
}
