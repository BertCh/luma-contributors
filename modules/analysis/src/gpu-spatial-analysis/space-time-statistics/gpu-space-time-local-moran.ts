// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryOutputRanges,
  getTrajectoryBindingViews
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocalPermutationTest,
  type GPULocalPermutationTestProps
} from '../permutation-inference/gpu-local-permutation-test';
import {
  validateGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';

const OPERATION = 'GPUSpaceTimeLocalMoran';

/**
 * Space-time local Moran recipes after GeoDa 1.22 (Anselin et al.), tested by conditional
 * permutation.
 *
 * - `'lagged'`: the bivariate local Moran of one variable at two times,
 *   `I_i = z_{i,t} sum_j w_ij z_{j,t-lag}`. `direction: 'inward'` (default, the GeoDa "inward
 *   influence" form) relates the focal present value to the neighbors' past values;
 *   `'outward'` relates the focal past value to the neighbors' present values.
 * - `'differential'`: the univariate local Moran of the change `d_i = y_{i,t} - y_{i,t-lag}`
 *   (GeoDa "Local Differential Moran's I"; GeoDa documents it as the Moran statistic of the
 *   difference variable).
 */
export type GPUSpaceTimeLocalMoranVariant = 'lagged' | 'differential';

/** Properties for {@link GPUSpaceTimeLocalMoran}. */
export type GPUSpaceTimeLocalMoranProps = Omit<
  GPULocalPermutationTestProps,
  'weights' | 'values' | 'secondValues' | 'statistic' | 'joinCase'
> & {
  /** Spatial weights `W` over the cells (`cellCount` rows), not the space-time weights. */
  weights: GPUSpatialWeights;
  /** Dense float32 cube, `cellCount * periodCount` values in `layout`. NaN marks a missing value. */
  cube: GraphDataView<'float32'>;
  /** Cube layout, see `GPUCubeLayoutTranspose`. */
  layout: GPUSpaceTimeWeightsLayout;
  /** Number of periods (compile-time). */
  periodCount: number;
  /** Period `t` analysed, an integer in `[lag, periodCount)` (compile-time). */
  period: number;
  /** Lag in periods, at least 1. Defaults to 1. */
  lag?: number;
  /** Recipe. */
  variant: GPUSpaceTimeLocalMoranVariant;
  /** Direction of `'lagged'`. Defaults to `'inward'`. Ignored by `'differential'`. */
  direction?: 'inward' | 'outward';
};

/**
 * Lagged and differential local Moran of one period of a space-time cube (STX-02). It slices the
 * cube into the period columns the GeoDa definitions use, then runs
 * {@link GPULocalPermutationTest}: `'localBivariateMoran'` with `values = y_t` and the permuted
 * `secondValues = y_{t-lag}` (or the reverse for `direction: 'outward'`), or `'localMoran'` of the
 * difference. Every output, the multiple-testing correction and the Philox streams are those of
 * the wrapped test, so the permutation null is conditional on the focal row and drawn over the
 * cells of the period.
 *
 * Caveat from GeoDa: the lagged bivariate form ignores in-situ correlation between the two
 * times, so High-High can reflect temporal inertia rather than diffusion. A row whose value or
 * lagged value is not finite is excluded from the analysis.
 *
 * Per-frame: the cube, weights, parameters and mask. Compile-time: everything else.
 */
export class GPUSpaceTimeLocalMoran implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeLocalMoranProps;

  constructor(props: GPUSpaceTimeLocalMoranProps) {
    this.id = props.id ?? 'space-time-local-moran';
    this.props = props;
    const id = this.id;
    const cellCount = validateGPUSpatialWeights(id, props.weights);
    validateCapacity(id, 'periodCount', props.periodCount);
    const lag = props.lag ?? 1;
    validateCapacity(id, 'lag', lag);
    validateCapacity(id, 'period', props.period, lag, props.periodCount - 1);
    validateSinglePackedView(props.cube, ['float32'], `${id} cube`);
    validateViewLength(id, 'cube', props.cube, cellCount * props.periodCount);
    validateCapacity(id, 'cube rows', cellCount * props.periodCount);
    validateTrajectoryOutputRanges(id, props.output);
    validateGraphOutputsDisjointFromInputs(
      id,
      Object.values(getTrajectoryBindingViews(props.output)),
      Object.values(getTrajectoryBindingViews({...props, output: undefined}))
    );
    if (props.layout !== 'period-major' && props.layout !== 'cell-major') {
      throw new Error(`${id} layout must be 'period-major' or 'cell-major'`);
    }
    if (props.variant !== 'lagged' && props.variant !== 'differential') {
      throw new Error(`${id} variant must be 'lagged' or 'differential'`);
    }
  }

  /** Returns the slice kernel and the wrapped permutation test's nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {props, id} = this;
    const {cube, layout, periodCount, period, variant, weights, ...testProps} = props;
    const cellCount = weights.offsets.length - 1;
    const lag = props.lag ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [cube]);
    const current = createTransientView(graph, `${id}-current`, 'float32', cellCount);
    const earlier = createTransientView(graph, `${id}-earlier`, 'float32', cellCount);
    const change = createTransientView(graph, `${id}-change`, 'float32', cellCount);
    const addressOf = (periodName: string) =>
      layout === 'period-major'
        ? `${periodName} * CELLS + index`
        : `index * PERIODS + ${periodName}`;
    const sliceNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-slice`,
      operation: OPERATION,
      variant: variant === 'differential' ? 'difference' : 'lagged',
      bindings: [
        {name: 'cube', view: cube, type: 'f32', access: 'read'},
        {name: 'current', view: current, type: 'f32', access: 'read_write'},
        {name: 'earlier', view: earlier, type: 'f32', access: 'read_write'},
        {name: 'change', view: change, type: 'f32', access: 'read_write'}
      ],
      invocationCount: cellCount,
      declarations: `const CELLS: u32 = ${cellCount}u;
const PERIODS: u32 = ${periodCount}u;
const PERIOD: u32 = ${period}u;
const LAG: u32 = ${lag}u;`,
      body: `let now = cube[cubeOffset + ${addressOf('PERIOD')}];
  let before = cube[cubeOffset + ${addressOf('(PERIOD - LAG)')}];
  current[currentOffset + index] = now;
  earlier[earlierOffset + index] = before;
  change[changeOffset + index] = now - before;`
    });
    const common = {...testProps, id: `${id}-test`, weights};
    let test: GPULocalPermutationTest;
    if (variant === 'differential') {
      test = new GPULocalPermutationTest({...common, values: change, statistic: 'localMoran'});
    } else {
      const outward = props.direction === 'outward';
      test = new GPULocalPermutationTest({
        ...common,
        values: outward ? earlier : current,
        secondValues: outward ? current : earlier,
        statistic: 'localBivariateMoran'
      });
    }
    return [sliceNode, ...test.getCommandNodes(graph)];
  }
}
