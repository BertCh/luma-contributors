// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialContext, GPUTemporalContext} from '../../analysis-core/contracts/index';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUKernelDensityKernelName} from './kernel-density-math';
import {
  GPUSpaceTimeKernelDensity,
  validateSpaceTimeKernelDensityBindings,
  type GPUSpaceTimeKernelDensityEdgeCorrection
} from './gpu-space-time-kernel-density';
import type {GPUSpaceTimeTemporalKernel} from './space-time-kernel-density-math';

import {
  stageDensityOutputViews,
  validateDensityViewBindings,
  validateDensityWritableViews
} from './density-point-passes';

const OPERATION = 'GPUSpaceTimeRelativeRisk';

/** Caller-owned outputs of {@link GPUSpaceTimeRelativeRisk}. */
export type GPUSpaceTimeRelativeRiskOutput = {
  /**
   * Risk per voxel, at least `grid.width * grid.height * sliceCount` float32 rows in the cell-major
   * order of `GPUSpaceTimeKernelDensity`. `'log'`: `log(f / g)`, `-Infinity` where the case density
   * is zero. `'ratio'`: `f / g`. NaN where either density is NaN or the control density is not
   * positive.
   */
  values: GraphDataView<'float32'>;
  /** Optional case probability density per voxel, same layout as `values`. */
  caseDensity?: GraphDataView<'float32'>;
  /** Optional control probability density per voxel, same layout as `values`. */
  controlDensity?: GraphDataView<'float32'>;
  /** Optional diagnostic scalars. */
  status?: {
    /** Optional one-row uint32 count of excluded case rows (see `GPUSpaceTimeKernelDensity`). */
    caseInvalidCount?: GraphDataView<'uint32'>;
    /** Optional one-row uint32 count of excluded control rows. */
    controlInvalidCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUSpaceTimeRelativeRisk}. Per-frame and compile-time props are those of
 * {@link GPUSpaceTimeKernelDensityProps}; both point sets share the grid, slices, bandwidths,
 * kernels, windows and index.
 */
export type GPUSpaceTimeRelativeRiskProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'space-time-relative-risk'`. */
  id?: string;
  /** As for `GPUSpaceTimeKernelDensity`. */
  spatialContext?: GPUSpatialContext;
  /** As for `GPUSpaceTimeKernelDensity`; both time columns must share it. */
  temporalContext?: GPUTemporalContext;
  /** Case positions. */
  casePositions: GraphDataView<'float32x2'>;
  /** Case times, relative float32 (see `GPUSpaceTimeKernelDensity`). */
  caseTimes: GraphDataView<'float32'>;
  /** Optional case weights. */
  caseWeights?: GraphDataView<'float32'>;
  /** Control positions. */
  controlPositions: GraphDataView<'float32x2'>;
  /** Control times, in the same unit and origin as the case times. */
  controlTimes: GraphDataView<'float32'>;
  /** Optional control weights. */
  controlWeights?: GraphDataView<'float32'>;
  /** Per-frame parameters written with `getGPUSpaceTimeKernelDensityParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Output raster grid. */
  grid: GPURasterMetadata;
  /** Number of time slices. */
  sliceCount: number;
  /** Dimensions of the internal grid indexes. */
  indexGridSize: readonly [number, number];
  /** Inclusive point domain of the internal grid indexes. */
  indexBounds: readonly [number, number, number, number];
  /** Spatial kernel. Defaults to `'gaussian'`. */
  spatialKernel?: GPUKernelDensityKernelName;
  /** Temporal kernel. Defaults to `'gaussian'`. */
  temporalKernel?: GPUSpaceTimeTemporalKernel;
  /** Spatial edge correction. Defaults to `'uniform'`. */
  spatialEdgeCorrection?: GPUSpaceTimeKernelDensityEdgeCorrection;
  /** Temporal edge correction. Defaults to `'uniform'`. */
  temporalEdgeCorrection?: GPUSpaceTimeKernelDensityEdgeCorrection;
  /** `'log'` (default, sparr `log = TRUE`) or `'ratio'`. Compile-time. */
  scale?: 'log' | 'ratio';
  /** Caller-owned outputs. */
  output: GPUSpaceTimeRelativeRiskOutput;
};

/**
 * Space-time relative risk with time-varying controls (sparr 2.3-16 `spattemp.risk`, Fernando and
 * Hazelton 2014): `rho(u, t) = f(u, t) / g(u, t)` where `f` and `g` are the case and control
 * space-time probability densities of {@link GPUSpaceTimeKernelDensity} (`normalization:
 * 'probability'`), computed with the same bandwidths, kernels, windows and grid, and reported as
 * `log(rho)` by default. Under equal case and control distributions `rho` is 1.
 *
 * Not implemented: time-static controls (a spatial-only control density with the conditional case
 * density), asymptotic tolerance contours and p-values, and adaptive or selected bandwidths. The
 * sparr conventions for those are not reproduced here and the sparr fixture is pending.
 *
 * The two densities are built as two `GPUSpaceTimeKernelDensity` pipelines in one graph (two
 * separate grid indexes), followed by one voxel kernel for the ratio.
 */
export class GPUSpaceTimeRelativeRisk implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeRelativeRiskProps;
  /** Resolved scale. */
  readonly scale: 'log' | 'ratio';

  constructor(props: GPUSpaceTimeRelativeRiskProps) {
    this.id = props.id ?? 'space-time-relative-risk';
    this.props = props;
    const {id} = this;
    this.scale = props.scale ?? 'log';
    validateRequirement(id, ['log', 'ratio'].includes(this.scale), 'scale must be log or ratio');
    validateSinglePackedView(props.casePositions, ['float32x2'], `${id} casePositions`);
    validateSinglePackedView(props.controlPositions, ['float32x2'], `${id} controlPositions`);
    validateRequirement(
      id,
      props.caseTimes.length === props.casePositions.length &&
        props.controlTimes.length === props.controlPositions.length,
      'times lengths must equal positions lengths'
    );
    // Validate both datasets (including optional weights and status) at construction. The caller's
    // risk view is only a metadata stand-in for an omitted intermediate density output.
    this.createDensity(
      'case',
      props.output.caseDensity ?? props.output.values,
      props.output.status?.caseInvalidCount
    );
    this.createDensity(
      'control',
      props.output.controlDensity ?? props.output.values,
      props.output.status?.controlInvalidCount
    );
    validateSinglePackedView(props.output.values, ['float32'], `${id} output.values`);
    for (const [name, view] of [
      ['output.caseDensity', props.output.caseDensity],
      ['output.controlDensity', props.output.controlDensity]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
        validateRequirement(
          id,
          view.length >= props.grid.width * props.grid.height * props.sliceCount,
          `${name} must hold grid.width * grid.height * sliceCount rows`
        );
      }
    }
    const outputViews = [
      props.output.values,
      props.output.caseDensity,
      props.output.controlDensity,
      props.output.status?.caseInvalidCount,
      props.output.status?.controlInvalidCount
    ].filter(view => view !== undefined);
    validateDensityWritableViews(id, {
      values: props.output.values,
      caseDensity: props.output.caseDensity,
      controlDensity: props.output.controlDensity,
      caseInvalidCount: props.output.status?.caseInvalidCount,
      controlInvalidCount: props.output.status?.controlInvalidCount
    });
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.casePositions,
      props.caseTimes,
      props.caseWeights,
      props.controlPositions,
      props.controlTimes,
      props.controlWeights,
      props.parameters
    ]);
  }

  /** Returns the case density nodes, the control density nodes and the ratio kernel, in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, scale} = this;
    const {output: callerOutput} = props;
    let output = callerOutput;
    validateGraphViewsBelongToGraph(id, graph, [
      props.casePositions,
      props.caseTimes,
      props.caseWeights,
      props.controlPositions,
      props.controlTimes,
      props.controlWeights,
      props.parameters,
      output.values,
      output.caseDensity,
      output.controlDensity,
      output.status?.caseInvalidCount,
      output.status?.controlInvalidCount
    ]);
    validateDensityViewBindings(
      graph,
      id,
      [
        ['casePositions', props.casePositions],
        ['caseTimes', props.caseTimes],
        ['caseWeights', props.caseWeights],
        ['controlPositions', props.controlPositions],
        ['controlTimes', props.controlTimes],
        ['controlWeights', props.controlWeights],
        ['parameters', props.parameters],
        ['output.values', output.values],
        ['output.caseDensity', output.caseDensity],
        ['output.controlDensity', output.controlDensity],
        ['output.status.caseInvalidCount', output.status?.caseInvalidCount],
        ['output.status.controlInvalidCount', output.status?.controlInvalidCount]
      ],
      {dispatchMetadata: false}
    );
    // Both nested index/sort/reduction/table layouts must fit before either pass allocates.
    validateSpaceTimeKernelDensityBindings(
      graph,
      this.createDensity(
        'case',
        output.caseDensity ?? output.values,
        output.status?.caseInvalidCount
      )
    );
    validateSpaceTimeKernelDensityBindings(
      graph,
      this.createDensity(
        'control',
        output.controlDensity ?? output.values,
        output.status?.controlInvalidCount
      )
    );
    const staged = stageDensityOutputViews(
      graph,
      id,
      OPERATION,
      {
        values: output.values,
        caseDensity: output.caseDensity,
        controlDensity: output.controlDensity,
        caseInvalidCount: output.status?.caseInvalidCount,
        controlInvalidCount: output.status?.controlInvalidCount
      },
      {
        values: props.grid.width * props.grid.height * props.sliceCount,
        caseDensity: props.grid.width * props.grid.height * props.sliceCount,
        controlDensity: props.grid.width * props.grid.height * props.sliceCount
      }
    );
    output = {
      ...staged.views,
      status: {
        caseInvalidCount: staged.views.caseInvalidCount,
        controlInvalidCount: staged.views.controlInvalidCount
      }
    };
    const voxelCount = props.grid.width * props.grid.height * props.sliceCount;
    const caseDensity =
      output.caseDensity ?? createTransientView(graph, `${id}-case-density`, 'float32', voxelCount);
    const controlDensity =
      output.controlDensity ??
      createTransientView(graph, `${id}-control-density`, 'float32', voxelCount);
    return [
      ...this.createDensity('case', caseDensity, output.status?.caseInvalidCount).getCommandNodes(
        graph
      ),
      ...this.createDensity(
        'control',
        controlDensity,
        output.status?.controlInvalidCount
      ).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ratio`,
        operation: OPERATION,
        variant: `ratio-${scale}`,
        bindings: [
          {name: 'caseDensity', view: caseDensity, type: 'f32', access: 'read'},
          {name: 'controlDensity', view: controlDensity, type: 'f32', access: 'read'},
          {name: 'risk', view: output.values, type: 'f32', access: 'read_write'}
        ],
        invocationCount: voxelCount,
        workgroupSize: 64,
        declarations: `${IS_FINITE_VALUE_WGSL}
fn getNaNValue() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
fn getNegativeInfinity() -> f32 { var bits = 0xff800000u; return bitcast<f32>(bits); }`,
        body: `let numerator = caseDensity[caseDensityOffset + index];
  let denominator = controlDensity[controlDensityOffset + index];
  let defined = isFiniteValue(numerator) && isFiniteValue(denominator) && denominator > 0.0;
  ${
    scale === 'log'
      ? 'let ratio = select(getNegativeInfinity(), log(max(numerator, 1e-38)) - log(max(denominator, 1e-38)), numerator > 0.0);'
      : 'let ratio = numerator / max(denominator, 1e-38);'
  }
  risk[riskOffset + index] = select(getNaNValue(), ratio, defined);`
      }),
      ...staged.publishNodes
    ];
  }

  private createDensity(
    name: 'case' | 'control',
    values: GraphDataView<'float32'>,
    invalidCount: GraphDataView<'uint32'> | undefined
  ): GPUSpaceTimeKernelDensity {
    const {id, props} = this;
    return new GPUSpaceTimeKernelDensity({
      id: `${id}-${name}`,
      spatialContext: props.spatialContext,
      temporalContext: props.temporalContext,
      positions: name === 'case' ? props.casePositions : props.controlPositions,
      times: name === 'case' ? props.caseTimes : props.controlTimes,
      weights: name === 'case' ? props.caseWeights : props.controlWeights,
      parameters: props.parameters,
      grid: props.grid,
      sliceCount: props.sliceCount,
      indexGridSize: props.indexGridSize,
      indexBounds: props.indexBounds,
      spatialKernel: props.spatialKernel,
      temporalKernel: props.temporalKernel,
      spatialEdgeCorrection: props.spatialEdgeCorrection,
      temporalEdgeCorrection: props.temporalEdgeCorrection,
      normalization: 'probability',
      output: {values, status: invalidCount ? {invalidCount} : undefined}
    });
  }
}
