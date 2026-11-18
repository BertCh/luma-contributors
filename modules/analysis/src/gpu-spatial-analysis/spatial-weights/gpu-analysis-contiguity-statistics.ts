// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUColumnQuantiles} from '../../dataframe-analysis/column-classification/gpu-column-quantiles';
import {getGPUColumnQuantilesParameterValues} from '../../dataframe-analysis/column-classification/column-quantiles-parameters';
import {
  GPUContiguityWeights,
  type GPUContiguityWeightsCapacityPlan,
  type GPUContiguityWeightsObservation,
  type GPUContiguityWeightsProps
} from './gpu-contiguity-weights';
import {
  GPUSpatialWeightsSummary,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT
} from './gpu-spatial-weights-summary';
import {
  getGPUSpatialWeightsStatusViews,
  getGPUSpatialWeightsViews,
  type GPUSpatialWeightsOutput
} from './spatial-weights';

const OPERATION = 'GPUAnalysisContiguityStatistics';

/** Quantile ranks and converted cardinalities must be exact float32 integers. */
export const GPU_ANALYSIS_CONTIGUITY_STATISTICS_MAXIMUM_FEATURE_COUNT = 2 ** 24;

/** Counts have exactly the layout used by the existing spatial weights summary. */
export const GPU_ANALYSIS_CONTIGUITY_STATISTICS_COUNTS = GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts;

/** Caller-owned outputs; a zero validity refuses an exact profile, including island claims. */
export type GPUAnalysisContiguityStatisticsOutput = {
  /**
   * Binary CSR, retained for reuse. All three status words are required for capacity recovery.
   * overflow reports final-slot truncation only; candidateOverflow reports intermediate truncation.
   * With candidateOverflow, requiredCount is a lower bound on the full neighbor requirement.
   */
  weights: GPUSpatialWeightsOutput & {
    status: {
      overflow: GraphDataView<'uint32'>;
      requiredCount: GraphDataView<'uint32'>;
      candidateOverflow: GraphDataView<'uint32'>;
    };
  };
  /** One active neighbor count per FEATURE, including islands (zero). */
  cardinality: GraphDataView<'uint32'>;
  /** At least six words in GPU_ANALYSIS_CONTIGUITY_STATISTICS_COUNTS order. */
  counts: GraphDataView<'uint32'>;
  /** One float32: linear median over all features, including islands. */
  median: GraphDataView<'float32'>;
  /** One uint32: 1 only when both the candidate list and final CSR fit. */
  validity: GraphDataView<'uint32'>;
};

/** Same topology and capacities as GPUContiguityWeights, with additional profile outputs. */
export type GPUAnalysisContiguityStatisticsProps = Omit<GPUContiguityWeightsProps, 'output'> & {
  output: GPUAnalysisContiguityStatisticsOutput;
};

/** Read-back evidence; only a valid, untruncated run supplies a summary. */
export type GPUAnalysisContiguityStatisticsResult = {
  overflow: boolean;
  candidateOverflow: boolean;
} & (
  | {exact: false}
  | {
      exact: true;
      slots: number;
      islands: number;
      minimum: number;
      median: number;
      maximum: number;
    }
);

/** Decodes scalar readback without treating truncated topology as evidence of islands. */
export function decodeGPUAnalysisContiguityStatistics(values: {
  counts: ArrayLike<number>;
  median: ArrayLike<number>;
  validity: ArrayLike<number>;
  overflow: ArrayLike<number>;
  candidateOverflow: ArrayLike<number>;
}): GPUAnalysisContiguityStatisticsResult {
  for (const name of ['counts', 'median', 'validity', 'overflow', 'candidateOverflow'] as const) {
    validateCapacity(OPERATION, `${name}.length`, values[name].length, name === 'counts' ? 6 : 1);
  }
  const overflow = values.overflow[0] !== 0;
  const candidateOverflow = values.candidateOverflow[0] !== 0;
  if (values.validity[0] === 0 || overflow || candidateOverflow) {
    return {exact: false, overflow, candidateOverflow};
  }
  const layout = GPU_ANALYSIS_CONTIGUITY_STATISTICS_COUNTS;
  return {
    exact: true,
    overflow,
    candidateOverflow,
    slots: values.counts[layout.slots],
    islands: values.counts[layout.isolates],
    minimum: values.counts[layout.minimumCardinality],
    median: values.median[0],
    maximum: values.counts[layout.maximumCardinality]
  };
}

/**
 * Composes GPUContiguityWeights, GPUSpatialWeightsSummary and GPUColumnQuantiles for AX profiles.
 * One criterion per instance; run queen and rook separately when both are wanted. This is exact
 * for the supplied float32 equality/snap topology, not a test of geometric coverage validity.
 * Every ring of every part contributes to its FEATURE row, including holes. Rook requires matching
 * edge endpoints; it does not detect overlapping edges with different vertex segmentation.
 *
 * Per-frame: polygon contents. Compile-time: criterion, snapTolerance, coordinateExtent and view
 * shapes/capacities. No metric distance is computed, so there is no spatialContext to resolve.
 * A positive snap requires coordinateExtent so the inherited float32 snap-cell limit is checked.
 * Empty datasets are rejected, as by GPUContiguityWeights; isolated and empty features are allowed.
 *
 * If either overflow status is nonzero, counts and cardinalities become 0xffffffff, median
 * becomes NaN and validity is 0. overflow reports only final-slot truncation; candidateOverflow
 * independently reports intermediate truncation.
 * CSR/status remain available for recovery through getWeightsCapacityPlan. There is no readback
 * inside the contributor and no persistent owned GPU resource; scratch belongs to the graph.
 * The median is rounded once to float32 (half-integers above 2^23 may round).
 */
export class GPUAnalysisContiguityStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUAnalysisContiguityStatisticsProps;
  private readonly contiguity: GPUContiguityWeights;

  constructor(props: GPUAnalysisContiguityStatisticsProps) {
    this.id = props.id ?? 'analysis-contiguity-statistics';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validateRequirement(
      id,
      Boolean(output.weights.status?.requiredCount && output.weights.status?.candidateOverflow),
      'output.weights.status must include requiredCount and candidateOverflow'
    );
    validateRequirement(
      id,
      !((props.snapTolerance ?? 0) > 0 && props.coordinateExtent === undefined),
      'coordinateExtent is required when snapTolerance is positive (maximum 2^24 snap cells)'
    );
    this.contiguity = new GPUContiguityWeights({...props, id, output: output.weights});
    const featureCount = this.contiguity.getFeatureCount();
    validateCapacity(
      id,
      'featureCount',
      featureCount,
      1,
      GPU_ANALYSIS_CONTIGUITY_STATISTICS_MAXIMUM_FEATURE_COUNT
    );
    for (const name of ['cardinality', 'counts', 'validity'] as const) {
      validateSinglePackedView(output[name], ['uint32'], `${id} output.${name}`);
    }
    validateSinglePackedView(output.median, ['float32'], `${id} output.median`);
    validateViewLength(id, 'output.cardinality', output.cardinality, featureCount);
    validateCapacity(id, 'output.counts.length', output.counts.length, 6);
    validateCapacity(id, 'output.median.length', output.median.length, 1);
    validateCapacity(id, 'output.validity.length', output.validity.length, 1);
    const outputs = this.getOutputViews();
    const inputs = this.getPolygonViews();
    for (let outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
      validateGraphOutputsDisjointFromInputs(
        id,
        [outputs[outputIndex]],
        [...inputs, ...outputs.slice(outputIndex + 1)]
      );
    }
  }

  /** Topology grow plan only; its memory estimate excludes the summary and quantile scratch. */
  getWeightsCapacityPlan(
    observation: GPUContiguityWeightsObservation = {}
  ): GPUContiguityWeightsCapacityPlan {
    return this.contiguity.getCapacityPlan(observation);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const featureCount = this.contiguity.getFeatureCount();
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.getPolygonViews(),
      ...this.getOutputViews()
    ]);
    for (const view of [...this.getPolygonViews(), ...this.getOutputViews()]) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'input/output binding',
          getViewBindingRange(view).size,
          'use smaller packed polygon and output views'
        );
      }
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'pairCapacity scratch',
      this.contiguity.getPairCapacity() * Uint32Array.BYTES_PER_ELEMENT,
      'use a smaller pairCapacity'
    );
    // Pair keys, pair sorting and scans each use one word per candidate. Vertex grouping's
    // terminal entry is the largest vertex-stage view. Check both before declaring any scratch.
    // Feature and slot scratch fit the already checked CSR offsets and neighbor views.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'vertex grouping scratch',
      (props.polygons.positions.length + 1) * Uint32Array.BYTES_PER_ELEMENT,
      'use fewer polygon vertices'
    );
    const statistics = createTransientView(graph, `${id}-statistics`, 'float32', 3);
    const values = createTransientView(graph, `${id}-cardinality-values`, 'float32', featureCount);
    const parameters = createTransientView(graph, `${id}-quantile-parameters`, 'float32', 5);
    const validCount = createTransientView(graph, `${id}-quantile-count`, 'uint32', 1);
    const parameterValues = getGPUColumnQuantilesParameterValues({quantiles: [0.5]});
    return [
      ...this.contiguity.getCommandNodes(graph),
      ...new GPUSpatialWeightsSummary({
        id: `${id}-summary`,
        weights: this.contiguity.outputWeights,
        output: {statistics, counts: output.counts, cardinality: output.cardinality}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-quantile-parameters`,
        operation: OPERATION,
        variant: 'quantile-parameters',
        bindings: [{name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}],
        invocationCount: 1,
        body: Array.from(
          parameterValues,
          (value, index) => `parameters[parametersOffset + ${index}u] = ${value.toFixed(1)};`
        ).join('\n')
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cardinality-values`,
        operation: OPERATION,
        variant: 'cardinality-values',
        bindings: [
          {name: 'cardinality', view: output.cardinality, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read_write'}
        ],
        invocationCount: featureCount,
        body: 'values[valuesOffset + index] = f32(cardinality[cardinalityOffset + index]);'
      }),
      ...new GPUColumnQuantiles({
        id: `${id}-median`,
        values,
        parameters,
        quantileCount: 1,
        output: {quantiles: output.median, validCount}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validity`,
        operation: OPERATION,
        variant: 'validity',
        bindings: [
          {name: 'overflow', view: output.weights.status.overflow, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.weights.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'cardinality', view: output.cardinality, type: 'u32', access: 'read_write'},
          {name: 'counts', view: output.counts, type: 'u32', access: 'read_write'},
          {name: 'median', view: output.median, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(featureCount, 6),
        declarations: GET_NAN_WGSL,
        body: `let valid = overflow[overflowOffset] == 0u && candidateOverflow[candidateOverflowOffset] == 0u;
  if (index == 0u) {
    validity[validityOffset] = select(0u, 1u, valid);
    if (!valid) { median[medianOffset] = getNaN(); }
  }
  if (!valid) {
    if (index < 6u) { counts[countsOffset + index] = 0xffffffffu; }
    if (index < ${featureCount}u) { cardinality[cardinalityOffset + index] = 0xffffffffu; }
  }`
      })
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      ...getGPUSpatialWeightsViews(output.weights),
      ...getGPUSpatialWeightsStatusViews(output.weights),
      output.cardinality,
      output.counts,
      output.median,
      output.validity
    ];
  }

  private getPolygonViews(): GraphDataView[] {
    const {polygons} = this.props;
    return [
      polygons.positions as GraphDataView<'float32x2'>,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets
    ];
  }
}
