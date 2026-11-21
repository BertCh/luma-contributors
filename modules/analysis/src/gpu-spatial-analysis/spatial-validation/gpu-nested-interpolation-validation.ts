// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateSinglePackedView,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validatePredictionViews} from '../../analysis-core/kernels/gpu-prediction-utils';
import {type GPUPredictionCovariance} from '../spatial-interpolation/gpu-local-kriging';
import type {GPULocalRBFProps} from '../spatial-interpolation/gpu-local-rbf';
import {getNestedValidationNodes} from './nested-validation-nodes';
import {snapshotNestedValidationProps} from './nested-validation-construction';
import {getGPULocalKrigingParameterValues} from '../spatial-interpolation/prediction-covariance';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT} from './nested-validation-limits';
export {GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT} from './nested-validation-limits';

/** Exact bounded neighborhood scans retain their original small reference envelope. */
export const GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT = 256;
/** Matches the local-kriging exhaustive-neighborhood limit; scores and metrics reduce row tiles. */
export const GPU_NESTED_VALIDATION_MAXIMUM_ROW_COUNT = 4096;
/** Eight includes a model upload, fit, mask/collection, and fixed graph overhead per split. */
const NESTED_FIT_COMMAND_NODE_ALLOWANCE = 8;
/** The existing cross-validation mask computes squared distances in float32. */
const NESTED_MAXIMUM_BUFFER_DISTANCE = 1e18;
/** Bounds the private dense neighborhood systems used by each candidate fit. */
export const GPU_NESTED_VALIDATION_MAXIMUM_NEIGHBOR_COUNT = 32;
/** Outer-fold expansion multiplies the full inner search command graph. */
const MAXIMUM_FOLD_COUNT = 16;
/** Each candidate is fitted separately for every inner split. */
const MAXIMUM_INNER_FOLD_COUNT = 8;

export type GPUNestedInterpolationCandidate =
  | {mode: 'ordinary-kriging'; covariance: GPUPredictionCovariance}
  | {
      mode: 'rbf';
      kernel?: GPULocalRBFProps['kernel'];
      degree?: GPULocalRBFProps['degree'];
      epsilon: number;
      smoothing?: number;
    };
export type GPUNestedInterpolationValidationProps = {
  id?: string;
  /** Tiled row scoring by default; reference retains the original 256-row envelope. */
  mode?: 'reference' | 'tiled';
  positions: GraphDataView<'float32x2'>;
  response: GraphDataView<'float32'>;
  /** Explicit outer fold masks, fold-major. Overlap is rejected by GPU status, never silently fitted. */
  training: GraphDataView<'uint32'>;
  testing: GraphDataView<'uint32'>;
  foldCount: number;
  /**
   * Inner block membership; only outer-training rows enter inner assignment and scoring. The
   * inner folds are dealt by `GPUSpatialCrossValidation` among the blocks that hold a training row
   * of the outer fold, so for one `seed` a block can be in different inner folds under different
   * outer folds (whenever an outer fold holds out whole inner blocks), and no inner fold is empty
   * while at least `innerFoldCount` blocks remain.
   */
  innerBlocks: GraphDataView<'uint32'>;
  innerFoldCount?: number;
  innerBufferDistance?: number;
  seed?: number;
  neighborCount?: number;
  /** Fixed hyperparameter ladder; model/range/sill/nugget or RBF inverse bandwidth is selected using inner held-out MSE. */
  candidates: readonly GPUNestedInterpolationCandidate[];
  output: {
    /** Per outer fold: chosen candidate, or 0xffffffff when no candidate covers every tested inner row. */
    selected: GraphDataView<'uint32'>;
    /** Outer-fold-major candidate scores [MSE, scored rows, missing predictions]. */
    scores: GraphDataView<'float32'>;
    /** Outer-fold-major predictions; only outer testing rows are published. */
    predictions: GraphDataView<'float32'>;
    /** Per fold [scored count, missing count, MSE, MAE, signed bias, R²]. */
    metrics: GraphDataView<'float32'>;
    /** Bit1 invalid overlapping outer masks, bit2 no selected candidate, bit4 missing heldout predictions, bit8 empty heldout set, bit16 nonfinite heldout metric arithmetic. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Resident exact nested search with tiled row scores/metrics and a bounded exhaustive neighbor
 * search. Outer test outcomes never affect inner fits or selection. Fold/split indices are runtime
 * parameters; all folds share the same fixed set of shader sources and pipelines.
 * Omitted mode and 'tiled' accept at most 4096 observations total; 'reference' accepts at most
 * 256 observations in one tile. Both routes accept 2–16 outer folds, 2–8 inner folds per outer
 * fold, 1–16 candidates per ladder, 2–32 neighbors per fit, and innerBufferDistance in [0, 1e18]
 * native planar units. Multi-tile routes use a tile width of at least 64 rows and merge at most
 * 64 metric partials per outer fold. Omitted mode retains one tile for at most 256 observations.
 * The estimated graph is limited to 32,768 command nodes.
 */
export class GPUNestedInterpolationValidation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUNestedInterpolationValidationProps;
  constructor(props: GPUNestedInterpolationValidationProps) {
    props = snapshotNestedValidationProps(props);
    this.props = props;
    this.id = props.id ?? 'nested-interpolation-validation';
    const rows = props.response.length;
    validateRequirement(
      this.id,
      props.mode === undefined || props.mode === 'reference' || props.mode === 'tiled',
      "mode must be 'reference' or 'tiled'"
    );
    validateCapacity(
      this.id,
      'rows',
      rows,
      3,
      props.mode === 'reference'
        ? GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT
        : GPU_NESTED_VALIDATION_MAXIMUM_ROW_COUNT
    );
    validateCapacity(this.id, 'foldCount', props.foldCount, 2, MAXIMUM_FOLD_COUNT);
    validateCapacity(
      this.id,
      'innerFoldCount',
      props.innerFoldCount ?? 3,
      2,
      MAXIMUM_INNER_FOLD_COUNT
    );
    validateCapacity(
      this.id,
      'candidates',
      props.candidates.length,
      1,
      GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT
    );
    validateCapacity(
      this.id,
      'neighborCount',
      props.neighborCount ?? 16,
      2,
      GPU_NESTED_VALIDATION_MAXIMUM_NEIGHBOR_COUNT
    );
    validateCapacity(this.id, 'seed', props.seed ?? 0, 0);
    validateFiniteNumber(
      this.id,
      'innerBufferDistance',
      props.innerBufferDistance ?? 0,
      0,
      NESTED_MAXIMUM_BUFFER_DISTANCE
    );
    for (const [index, candidate] of props.candidates.entries()) {
      const name = `candidates[${index}]`;
      if (candidate.mode === 'ordinary-kriging') {
        try {
          const parameters = getGPULocalKrigingParameterValues({covariance: candidate.covariance});
          parameters.forEach((value, slot) =>
            validateFiniteNumber(this.id, `${name}.covariance parameter ${slot}`, value)
          );
        } catch (error) {
          throw new Error(`${this.id} ${name} ${String(error)}`);
        }
      } else {
        validateRequirement(
          this.id,
          candidate.mode === 'rbf',
          `${name}.mode must be ordinary-kriging or rbf`
        );
        const kernel = candidate.kernel ?? 'thin-plate-spline';
        const degree =
          candidate.degree ??
          (kernel === 'quintic'
            ? 2
            : ['thin-plate-spline', 'cubic'].includes(kernel)
              ? 1
              : ['linear', 'multiquadric'].includes(kernel)
                ? 0
                : -1);
        validateRequirement(
          this.id,
          [
            'thin-plate-spline',
            'cubic',
            'gaussian',
            'linear',
            'quintic',
            'multiquadric',
            'inverse-multiquadric',
            'inverse-quadratic'
          ].includes(kernel),
          `${name}.kernel must be a supported RBF kernel`
        );
        validateRequirement(
          this.id,
          [-1, 0, 1, 2].includes(degree) &&
            (!['thin-plate-spline', 'cubic'].includes(kernel) || degree >= 1) &&
            (!['linear', 'multiquadric'].includes(kernel) || degree >= 0) &&
            (kernel !== 'quintic' || degree === 2),
          `${name}.degree must match the RBF polynomial requirement`
        );
        validateFiniteNumber(this.id, `${name}.epsilon`, candidate.epsilon, 0);
        validateRequirement(
          this.id,
          Math.fround(candidate.epsilon) > 0,
          `${name}.epsilon must be positive in float32`
        );
        validateFiniteNumber(this.id, `${name}.smoothing`, candidate.smoothing ?? 0, 0);
      }
    }
    this.estimatedCommandNodeCount =
      1 +
      props.foldCount *
        (32 +
          NESTED_FIT_COMMAND_NODE_ALLOWANCE *
            props.candidates.length *
            ((props.innerFoldCount ?? 3) + 1));
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    for (const [view, format, length] of [
      [props.positions, 'float32x2', rows],
      [props.response, 'float32', rows],
      [props.training, 'uint32', rows * props.foldCount],
      [props.testing, 'uint32', rows * props.foldCount],
      [props.innerBlocks, 'uint32', rows],
      [props.output.selected, 'uint32', props.foldCount],
      [props.output.scores, 'float32', props.foldCount * props.candidates.length * 3],
      [props.output.predictions, 'float32', rows * props.foldCount],
      [props.output.metrics, 'float32', 6 * props.foldCount],
      [props.output.status, 'uint32', props.foldCount]
    ] as const) {
      validateSinglePackedView(view, [format], this.id);
      if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
    }
    validateRegressionOutputDispatches(this.id, props.output, []);
    validatePredictionViews(
      this.id,
      [props.positions, props.response, props.training, props.testing, props.innerBlocks],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getNestedValidationNodes(graph, this.id, this.props);
  }
}
