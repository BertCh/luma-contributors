// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryOutputRanges,
  validateTrajectoryWritableDispatches,
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
  validateGraphPorts,
  validateOption,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocalPermutationTest,
  validateLocalPermutationSettings,
  type GPULocalPermutationTestProps
} from '../permutation-inference/gpu-local-permutation-test';
import {sliceGraphChunkGroupView} from '../../analysis-core/kernels/graph-chunk-groups';
import {validatePermutationInputBindings} from '../permutation-inference/permutation-inference-kernels';
import {snapshotLocalPermutationProps} from '../permutation-inference/permutation-construction';
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
    props = snapshotLocalPermutationProps(props, [
      'id',
      'weights',
      'mask',
      'alternative',
      'islands',
      'parameters',
      'maximumPermutations',
      'stopAfterExceedances',
      'maximumNeighbors',
      'extendedNeighborScratch',
      'rowTileSize',
      'multipleTesting',
      'output',
      'cube',
      'layout',
      'periodCount',
      'period',
      'lag',
      'variant',
      'direction'
    ]);
    this.id = props.id ?? 'space-time-local-moran';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      props.output.classes === undefined,
      'output.classes applies only to localGeary; use output.quadrants for lagged Moran or GPULocalPermutationTest with statistic localGeary for classes'
    );
    validateRequirement(
      id,
      props.output.quadrants === undefined || props.variant === 'lagged',
      'output.quadrants requires lagged Moran; use variant lagged or omit output.quadrants for differential Moran'
    );
    validateGraphPorts(id, props, {
      cube: {required: true, viewType: 'data', formats: ['float32']},
      parameters: {required: true, viewType: 'data', formats: ['uint32']}
    });
    const cellCount = validateGPUSpatialWeights(id, props.weights);
    validateCapacity(id, 'periodCount', props.periodCount);
    const lag = props.lag ?? 1;
    validateCapacity(id, 'lag', lag);
    validateCapacity(id, 'period', props.period, lag, props.periodCount - 1);
    validateSinglePackedView(props.cube, ['float32'], `${id} cube`);
    validateViewLength(id, 'cube', props.cube, cellCount * props.periodCount);
    validateCapacity(id, 'cube rows', cellCount * props.periodCount);
    validateTrajectoryOutputRanges(id, props.output);
    validateTrajectoryWritableDispatches(id, [
      {overflow: props.output.status.overflow, quadrants: props.output.quadrants}
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      Object.values(getTrajectoryBindingViews(props.output)),
      Object.values(getTrajectoryBindingViews({...props, output: undefined}))
    );
    validateOption(id, 'layout', props.layout, ['period-major', 'cell-major']);
    validateOption(id, 'variant', props.variant, ['lagged', 'differential']);
    if (props.direction !== undefined)
      validateOption(id, 'direction', props.direction, ['inward', 'outward']);
    // A read-only prefix has the delegated row shape without allocating slice scratch.
    // Run the actual child's construction checks, including every optional output/setting.
    validateLocalPermutationSettings(id, props);
    // Validation-only prefix: public data views are immutable descriptors. Keep the buffer
    // identity and packed layout while checking the delegated row shape without graph methods.
    const values: GraphDataView<'float32'> = {...props.cube, length: cellCount};
    new GPULocalPermutationTest({
      ...props,
      id,
      values,
      statistic: props.variant === 'differential' ? 'localMoran' : 'localBivariateMoran',
      secondValues: props.variant === 'differential' ? undefined : values
    });
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
    const values = sliceGraphChunkGroupView(graph, cube, 0, cellCount);
    validatePermutationInputBindings(
      graph,
      {
        ...props,
        id,
        operation: OPERATION,
        values,
        secondValues: variant === 'lagged' ? values : undefined,
        analysisCount: props.output.analysisCount,
        islandCount: props.output.islandCount
      },
      [
        props.output.exceedances,
        props.output.analysisCount,
        props.output.permutationCounts,
        props.output.pseudoPValues,
        props.output.observed,
        props.output.significant,
        props.output.adjustedPValues,
        props.output.classes,
        props.output.quadrants,
        props.output.islandCount,
        ...Object.values(props.output.status)
      ]
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'space-time slice scratch',
      cellCount * 4,
      'reduce cellCount or request larger device bindings'
    );
    if (variant === 'lagged' || props.stopAfterExceedances !== undefined)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'two-column local inference scratch',
        cellCount * 8,
        'reduce cellCount or request larger device bindings'
      );
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
