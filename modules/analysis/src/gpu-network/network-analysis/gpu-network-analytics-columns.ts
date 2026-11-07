// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createMaskedWeakComponentNodes} from '../network-masked-weak-components';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkAdjacency
} from '../../analysis-core/network-topology';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/index';

/**
 * One node-aligned analytics column and its optional normalized and extent companions.
 *
 * @typeParam Format Element format of the raw column.
 */
export type GPUNetworkAnalyticsColumn<Format extends 'uint32' | 'float32'> = {
  /** Caller-owned raw column with one packed row per node. Any graph view, including transients. */
  output: GraphDataView<Format>;
  /**
   * Optional caller-owned column holding `(value - min) / (max - min)`, or 0 when
   * `max <= min`. Adding it schedules a GPU min/max reduction. Any graph view of one packed
   * `float32` row per node.
   */
  normalized?: GraphDataView<'float32'>;
  /**
   * Optional two-row `[min, max]` of the raw column. Graph-owned scratch when omitted but
   * `normalized` is given.
   */
  extent?: GraphDataView<Format>;
};

/**
 * Properties for {@link GPUNetworkAnalyticsColumns}.
 *
 * Compile-time: node count, which metrics and optional views exist, and `maximumIterations`,
 * which is baked into the command graph.
 * Per-frame: the contents of the CSR, so closing a road in a live buffer changes every column on
 * the next encoding without recompiling.
 */
export type GPUNetworkAnalyticsColumnsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-analytics-columns'`. */
  id?: string;
  /**
   * Forward network connectivity; edge weights are ignored. `offsets` has `nodeCount + 1` rows and
   * `neighbors` indices `>= nodeCount` are ignored. Any graph view, so a CSR built in the same
   * graph by `GPUCOOToCSR` works as well as an imported buffer.
   *
   * Without `reverseTopology` the graph is treated as UNDIRECTED, so the forward CSR must be
   * symmetric: every two-way road is listed in both directions (the same convention as
   * `GPUNetworkReachability`). With `reverseTopology` the graph is directed.
   */
  topology: GPUNetworkAdjacency;
  /** Optional reverse CSR with the same node count. Its presence makes the graph directed. */
  reverseTopology?: Pick<GPUNetworkAdjacency, 'offsets' | 'neighbors'>;
  /** Outgoing degree (incident degree when undirected). */
  degree?: GPUNetworkAnalyticsColumn<'uint32'>;
  /** Incoming degree. Requires the reverse CSR. */
  inDegree?: GPUNetworkAnalyticsColumn<'uint32'>;
  /**
   * Weak connected component label, the lowest node index of each component once converged.
   * `maximumIterations` (default 32) is compile-time.
   */
  components?: {
    output: GraphDataView<'uint32'>;
    maximumIterations?: number;
    /** Caller-owned one-row status words. */
    status?: Pick<GPUIterationStatusPort, 'converged'>;
  };
};

type AnalyticsAlgorithm = {
  getNodes<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCommandNode<Parameters>[];
};

type AnalyticsMetric = {
  /** Node ID suffix, for example `page-rank`. */
  step: string;
  algorithm: AnalyticsAlgorithm;
  format?: 'uint32' | 'float32';
  raw?: GraphDataView<'uint32' | 'float32'>;
  normalized?: GraphDataView<'float32'>;
  extent?: GraphDataView<'uint32' | 'float32'>;
};

/**
 * Publishes node-aligned `float32` and `uint32` analytics columns for a CSR road network, ready
 * for deck.gl styling: degree, in-degree and weak component labels, optionally normalized to
 * [0, 1] with a GPU min/max.
 *
 * It runs analysis-owned kernels directly on the caller's CSR views, so nothing is rebuilt or
 * copied and transient CSRs (for example from `GPUCOOToCSR`) work. Published luma.gl graph
 * algorithms (PageRank, k-core, label propagation) only accept physical topology, so those
 * metrics are not offered. Every encoding recomputes every column.
 *
 * Without a reverse CSR the graph is undirected and the forward CSR must be symmetric. Reverse
 * CSRs can be built with `GPUCOOToCSR`.
 *
 * Non-goals: weighted metrics (edge weights are ignored), betweenness or closeness centrality,
 * modularity-optimizing communities, and CPU readback of any column.
 */
export class GPUNetworkAnalyticsColumns implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkAnalyticsColumnsProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  private readonly metrics: readonly AnalyticsMetric[];

  constructor(props: GPUNetworkAnalyticsColumnsProps) {
    this.id = props.id ?? 'network-analytics-columns';
    this.props = props;
    const {id} = this;
    this.nodeCount = validateGPUNetworkTopology(id, props.topology).nodeCount;
    const {nodeCount} = this;
    if (props.reverseTopology) {
      validatePackedView(
        props.reverseTopology.offsets,
        ['uint32'],
        `${id} reverseTopology.offsets`
      );
      validatePackedView(
        props.reverseTopology.neighbors,
        ['uint32'],
        `${id} reverseTopology.neighbors`
      );
      if (props.reverseTopology.offsets.length !== nodeCount + 1) {
        throw new Error(
          `${id} reverseTopology.offsets must contain the same number of rows as topology.offsets`
        );
      }
    }
    if (!props.degree && !props.inDegree && !props.components) {
      throw new Error(`${id} requires at least one metric`);
    }
    validateCapacity(
      id,
      'components iterations',
      props.components?.maximumIterations ?? 32,
      1,
      1024
    );
    if (props.inDegree && !props.reverseTopology) {
      throw new Error(`${id} inDegree requires reverseTopology`);
    }
    const nodeColumns: [string, GraphDataView, 'uint32' | 'float32'][] = [];
    const scalars: [string, GraphDataView | undefined, 'uint32' | 'float32'][] = [];
    const companions: [string, GraphDataView<'float32'> | GraphDataView | undefined, number][] = [];
    for (const [name, metric, format] of [
      ['degree', props.degree, 'uint32'],
      ['inDegree', props.inDegree, 'uint32']
    ] as const) {
      if (!metric) {
        continue;
      }
      nodeColumns.push([`${name}.output`, metric.output, format]);
      if (metric.normalized) {
        companions.push([`${name}.normalized`, metric.normalized, nodeCount]);
        validatePackedView(metric.normalized, ['float32'], `${id} ${name}.normalized`);
      }
      if (metric.extent) {
        companions.push([`${name}.extent`, metric.extent, 2]);
        validatePackedView(metric.extent, [format], `${id} ${name}.extent`);
      }
      if (metric.extent && !metric.normalized) {
        throw new Error(`${id} ${name}.extent requires ${name}.normalized`);
      }
    }
    if (props.components) {
      nodeColumns.push(['components.output', props.components.output, 'uint32']);
    }
    scalars.push(['components.status.converged', props.components?.status?.converged, 'uint32']);
    for (const [name, view, format] of nodeColumns) {
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must equal the node count`);
      }
    }
    for (const [name, view, format] of scalars) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    for (const [name, view, length] of companions) {
      if (view && view.length !== length) {
        throw new Error(
          `${id} ${name} must contain ${length === 2 ? 'exactly two rows' : 'one row per node'}`
        );
      }
    }
    const inputBuffers = [
      ...getGPUNetworkTopologyViews(props.topology),
      props.reverseTopology?.offsets,
      props.reverseTopology?.neighbors
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const outputBuffers = [
      ...nodeColumns.map(([, view]) => view),
      ...scalars.map(([, view]) => view),
      ...companions.map(([, view]) => view)
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (
      new Set(outputBuffers).size !== outputBuffers.length ||
      outputBuffers.some(buffer => inputBuffers.includes(buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers from each other and from inputs`);
    }

    this.metrics = this.createMetrics();
  }

  /** Returns each requested gpu-graph algorithm followed by the optional normalization nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount} = this;
    validateGraphViewsBelongToGraph(id, graph, getViews(props));
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (const metric of this.metrics) {
      nodes.push(...metric.algorithm.getNodes(graph));
    }
    for (const metric of this.metrics) {
      if (!metric.normalized || !metric.raw || !metric.format) {
        continue;
      }
      const extent =
        metric.extent ??
        createTransientView(graph, `${id}-${metric.step}-extent-scratch`, metric.format, 2);
      nodes.push(
        ...new GPUReduction({
          id: `${id}-${metric.step}-extent`,
          input: metric.raw,
          output: extent,
          operation: 'extent'
        }).getCommandNodes(graph)
      );
      const elementType = metric.format === 'uint32' ? 'u32' : 'f32';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${metric.step}-normalize`,
          operation: 'GPUNetworkAnalyticsColumns',
          variant: 'normalize',
          record: {datum: {keySpace: 'node'}, formula: 'gpu-network.analytics.normalize'},
          bindings: [
            {name: 'raw', view: metric.raw, type: elementType, access: 'read'},
            {name: 'extent', view: extent, type: elementType, access: 'read'},
            {name: 'normalized', view: metric.normalized, type: 'f32', access: 'read_write'}
          ],
          invocationCount: nodeCount,
          body: `let lowest = f32(extent[extentOffset]);
  let highest = f32(extent[extentOffset + 1u]);
  let range = highest - lowest;
  var scaled = 0.0;
  if (range > 0.0) {
    scaled = (f32(raw[rawOffset + index]) - lowest) / range;
  }
  normalized[normalizedOffset + index] = scaled;`
        })
      );
    }
    return nodes;
  }

  private createMetrics(): AnalyticsMetric[] {
    const {id, props, nodeCount} = this;
    const metrics: AnalyticsMetric[] = [];
    const createDegreeAlgorithm = (
      step: string,
      offsets: GraphDataView<'uint32'>,
      output: GraphDataView<'uint32'>
    ): AnalyticsAlgorithm => ({
      getNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: 'GPUNetworkAnalyticsColumns',
          variant: step,
          bindings: [
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'degrees', view: output, type: 'u32', access: 'read_write'}
          ],
          invocationCount: nodeCount,
          body: 'degrees[degreesOffset + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];'
        })
      ]
    });
    if (props.degree) {
      metrics.push({
        step: 'degree',
        algorithm: createDegreeAlgorithm('degree', props.topology.offsets, props.degree.output),
        format: 'uint32',
        raw: props.degree.output,
        normalized: props.degree.normalized,
        extent: props.degree.extent
      });
    }
    if (props.inDegree && props.reverseTopology) {
      metrics.push({
        step: 'in-degree',
        algorithm: createDegreeAlgorithm(
          'in-degree',
          props.reverseTopology.offsets,
          props.inDegree.output
        ),
        format: 'uint32',
        raw: props.inDegree.output,
        normalized: props.inDegree.normalized,
        extent: props.inDegree.extent
      });
    }
    if (props.components) {
      const {components} = props;
      metrics.push({
        step: 'components',
        algorithm: {
          getNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) =>
            createMaskedWeakComponentNodes<Parameters>(graph, {
              id: `${id}-components`,
              operation: 'GPUNetworkAnalyticsColumns',
              nodeCount,
              offsets: props.topology.offsets,
              neighbors: props.topology.neighbors,
              labels: components.output,
              converged: components.status?.converged,
              iterations: components.maximumIterations ?? 32
            })
        }
      });
    }
    return metrics;
  }
}

/** Returns every graph view referenced by the props. */
function getViews(props: GPUNetworkAnalyticsColumnsProps): (GraphDataView | undefined)[] {
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    props.reverseTopology?.offsets,
    props.reverseTopology?.neighbors,
    ...[props.degree, props.inDegree].flatMap(metric => [
      metric?.output,
      metric?.normalized,
      metric?.extent
    ]),
    props.components?.output,
    props.components?.status?.converged
  ];
}
