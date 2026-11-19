var e={kernelDensity:`gpu-spatial-analysis.kernel-density`,localMoran:`gpu-spatial-analysis.local-moran`,serviceAreas:`gpu-network.network-service-areas`},t={scenarios:[e.kernelDensity],examples:{kernelDensity:`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

// example:hide-start
declare const device: Device;
/** Observations as x, y pairs in metres of a local frame. */
declare const observations: Float32Array;
/** Study extent in the same metres, a whole number of cells. */
declare const extent: readonly [number, number, number, number];
// example:hide-end
// 1. Buffers: your data and your output, in GPU memory you own.
const [minX, minY, maxX, maxY] = extent;
const cellSize = 125; // metres
const [width, height] = [(maxX - minX) / cellSize, (maxY - minY) / cellSize].map(Math.round);
const pointBuffer = device.createBuffer({usage: Buffer.STORAGE, data: observations});
const densityBuffer = device.createBuffer({
  usage: Buffer.STORAGE | Buffer.COPY_SRC, // COPY_SRC: read the result back
  byteLength: width * height * 4
});

// 2. Views: tell the graph what the bytes are.
const graph = new GPUCommandGraph<void>(device, {id: 'kernel-density'});
const positions = graph.createDataView(
  graph.importBuffer(
    {id: 'observations', byteLength: pointBuffer.byteLength, usage: pointBuffer.usage},
    pointBuffer
  ),
  {format: 'float32x2', length: observations.length / 2}
);
const density = graph.createDataView(
  graph.importBuffer(
    {id: 'density', byteLength: densityBuffer.byteLength, usage: densityBuffer.usage},
    densityBuffer
  ),
  {format: 'float32', length: width * height}
);

// 3. Build: one operation adds all of its kernels.
const grid = makeGPURasterMetadata({width, height, cellSize, origin: [minX, maxY]});
const values = getGPUKernelDensityParameterValues({bandwidth: 400}, grid);
const parameters = new GPUParameterBuffer(device, {
  id: 'kde-parameters',
  format: 'float32',
  length: values.length,
  values
});
graph.add(
  new GPUKernelDensity({
    positions,
    parameters: parameters.importToGraph(graph),
    grid,
    indexGridSize: [32, 32],
    indexBounds: extent,
    output: {values: density}
  })
);

// 4. Compile once: order the kernels, create the pipelines.
const compiled = await graph.compileAsync();

// 5. Encode and submit: this is all a frame costs.
const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());

// 6. Read back, only if the CPU needs the numbers. A layer can draw densityBuffer directly.
const bytes = await densityBuffer.readAsync();
const observationsPerSquareMetre = new Float32Array(bytes.buffer, bytes.byteOffset, width * height);

// A new bandwidth is a 36-byte write. Nothing recompiles.
parameters.write(getGPUKernelDensityParameterValues({bandwidth: 250}, grid));
// example:hide-start
export {observationsPerSquareMetre};
// example:hide-end
`}},n={timeWindow:`gpu-dataframe.time-window-filter`,kernelDensity:`gpu-spatial-analysis.kernel-density`},r={scenarios:Object.values(n),examples:{}},i=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUContiguityWeights,
  GPUSpatialWeightsTransform,
  GPULocalMoran,
  type GPUContiguityWeightsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Uploaded tract geometry and poverty percentages; the caller owns the recording graph. */
export type LocalMoranInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'];
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Local Moran result figure. */
export async function runExample(device: Device, input: LocalMoranInput) {
  const {graph, tracts, values, parameters} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const count = values.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string, length = count) =>
    createTransientView(graph, name, 'float32', length, usage);
  const createInteger = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length, usage);
  // 16K slots and 64K candidate pairs comfortably cover this 791-tract topology.
  const weightOutput = {
    offsets: createInteger('weightsRowOffsets', count + 1),
    neighbors: createInteger('weightsNeighbors', 16384),
    weights: createFloat('weightsValues', 16384),
    status: {overflow: createInteger('weightsOverflow', 1)}
  };
  const output = {
    localI: createFloat('localI'),
    zScores: createFloat('zScores'),
    spatialLag: createFloat('spatialLag'),
    quadrants: createInteger('quadrants'),
    pValues: createFloat('pValues'),
    adjustedPValues: createFloat('adjustedPValues'),
    neighborCounts: createInteger('neighborCounts'),
    globalStatistics: createFloat('globalStatistics', 5)
  };
  // #region example
  const queen = new GPUContiguityWeights({
    polygons: tracts,
    criterion: 'queen',
    output: weightOutput
  });
  graph.add(queen);
  const rowWeights = new GPUSpatialWeightsTransform({
    weights: queen.outputWeights,
    operation: 'row'
  });
  graph.add(rowWeights);
  const moran = new GPULocalMoran({
    values,
    weights: rowWeights.outputWeights,
    parameters,
    islands: 'exclude',
    minimumNeighbors: 4,
    multipleTesting: 'benjamini-hochberg',
    quadrantGating: 'analytic',
    output
  });
  graph.add(moran);
  const {localI, adjustedPValues} = output;
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    weights: rowWeights.outputWeights,
    views: {
      values,
      ...output,
      localI,
      adjustedPValues,
      weightsRowOffsets: weightOutput.offsets,
      weightsNeighbors: weightOutput.neighbors,
      weightsValues: weightOutput.weights,
      weightsOverflow: weightOutput.status.overflow
    }
  };
}
`,a=`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// example:hide-start
// Abridged from modules/analysis/src/dataframe-analysis/time-window-filter/gpu-time-window-filter.ts
// (the float32 path without extra predicates) and time-window-classify-node.ts (the classify
// bindings). Only the WGSL text and the setup are hidden.
// example:hide-end

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUVisibilityWorkflow} from '@luma.gl/graph/gpu-core';
import {
  createPublishNode,
  createWGSLKernelNode,
  type GPUCompactOutput
} from '@luma-contributors/analysis/analysis-core';

// example:hide-start
declare const graph: GPUCommandGraph<void>;
declare const timeWindow: GraphDataView<'float32'>;
declare const timestamps: GraphDataView<'float32'>;
declare const timestampsLow: GraphDataView<'float32'>;
declare const outputMask: GraphDataView<'uint32'>;
declare const total: GraphDataView<'uint32'>;
declare const output: Omit<GPUCompactOutput, 'ids'> & {rows: GraphDataView<'uint32'>};
declare const classifyDeclarations: string;
declare const classifyBody: string;
// example:hide-end
// GPUTimeWindowFilter, abridged: the kernels it adds, in the order it adds them.
const nodes = [
  // classify: one invocation per row writes 1 or 0 into outputMask.
  createWGSLKernelNode<void>(graph, {
    id: 'time-window-filter-classify',
    operation: 'GPUTimeWindowFilter',
    variant: 'classify',
    bindings: [
      {name: 'params', view: timeWindow, type: 'f32', access: 'read'},
      {name: 'startHigh', view: timestamps, type: 'f32', access: 'read'},
      {name: 'startLow', view: timestampsLow, type: 'f32', access: 'read'},
      {name: 'outputMask', view: outputMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: timestamps.length,
    declarations: classifyDeclarations,
    body: classifyBody
  }),
  // compaction (luma.gl): reads outputMask, scans it, scatters the kept row indices.
  ...new GPUVisibilityWorkflow({
    id: 'time-window-filter-visibility',
    predicates: [{kind: 'time-range', mask: outputMask}],
    output: output.rows,
    count: total,
    outputMask
  }).getCommandNodes(graph),
  // publish: clamps the count into the caller's status.
  createPublishNode<void>(graph, {
    id: 'time-window-filter-publish',
    operation: 'GPUTimeWindowFilter',
    requiredCount: total,
    output: {ids: output.rows, status: output.status}
  })
];
// No edge and no order anywhere: the graph reads them off the bindings.
graph.add(nodes);
`,o={timeWindowFilter:`gpu-dataframe.time-window-filter`,classBreaks:`gpu-dataframe.class-breaks`,hilbertKeys:`gpu-spatial-analysis.hilbert-keys`,localMoran:e.localMoran},s={scenarios:Object.values(o),examples:{declareKernels:a,localMoranChain:i}},c={scenarios:[],examples:{serviceAreas:`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkServiceAreas,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs: the road network and the facilities' prepared nearest intersections. */
export type NetworkServiceAreasExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  facilityNodes: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkServiceAreasExampleInput) {
  const {graph, roads, facilityNodes} = input;
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const serviceAreas = new GPUNetworkServiceAreas({
    topology: roads,
    facilities: facilityNodes,
    maximumIterations: 24,
    output: {
      assignments: createTransientView(graph, 'assignments', 'uint32', nodeCount),
      costs: createTransientView(graph, 'costs', 'float32', nodeCount),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  });
  graph.add(serviceAreas);
  const {assignments, costs} = serviceAreas.props.output;
  // #endregion example
  return {
    serviceAreas,
    assignments,
    costs: costs!,
    converged: serviceAreas.props.output.status!.converged!
  };
}
`}},l={scenarios:[e.kernelDensity,e.localMoran],examples:{"buffer-views":`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

// example:hide-start
declare const device: Device;
/** Observations as x, y pairs. */
declare const observations: Float32Array;
/** Ring-to-vertex offsets of a polygon layer, with one terminal entry. */
declare const ringOffsets: Uint32Array;
// example:hide-end
const graph = new GPUCommandGraph<void>(device, {id: 'data-views'});

// Your buffer: the graph imports it and never destroys it.
const pointBuffer = device.createBuffer({usage: Buffer.STORAGE, data: observations});
const pointHandle = graph.importBuffer(
  {id: 'observations', byteLength: pointBuffer.byteLength, usage: pointBuffer.usage},
  pointBuffer
);

// A view lays a format over the bytes: two float32 per row, one row per point.
const positions = graph.createDataView(pointHandle, {
  format: 'float32x2',
  length: observations.length / 2
});

// Offsets are an ordinary view too: one uint32 per entry.
const offsetBuffer = device.createBuffer({usage: Buffer.STORAGE, data: ringOffsets});
const ringOffsetView = graph.createDataView(
  graph.importBuffer(
    {id: 'ring-offsets', byteLength: offsetBuffer.byteLength, usage: offsetBuffer.usage},
    offsetBuffer
  ),
  {format: 'uint32', length: ringOffsets.length}
);
// example:hide-start
export {positions, ringOffsetView};
// example:hide-end
`}},u={scenarios:[],examples:{}},d={scenarios:[],examples:{reachability:`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkReachability,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs; structureOnly selects synchronous compilation without execution. */
export type NetworkReachabilityExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  startNodes: GraphDataView<'uint32'>;
  activeIterations: GraphDataView<'uint32'>;
  costBreaks: GraphDataView<'float32'>;
  localIterations?: number;
  structureOnly?: boolean;
  /** Adds observation nodes after the contributor and before compilation. */
  beforeCompile?: (built: {
    output: NetworkReachabilityOutput;
    commandNodes: readonly GPUCommandNode[];
  }) => void;
};

/** Output views of the reachability example. */
export type NetworkReachabilityOutput = {
  costs: GraphDataView<'float32'>;
  bands: GraphDataView<'uint32'>;
  predecessors: GraphDataView<'uint32'>;
  bandCounts: GraphDataView<'uint32'>;
  status: {converged: GraphDataView<'uint32'>; iterationCount: GraphDataView<'uint32'>};
};

/** Builds the same public contributor shown in the card; submission belongs to the lens. */
export async function runExample(_device: Device, input: NetworkReachabilityExampleInput) {
  const {graph, roads, startNodes, activeIterations, costBreaks, localIterations} = input;
  const id = 'reachability';
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const output = {
    costs: createTransientView(graph, 'costs', 'float32', nodeCount),
    bands: createTransientView(graph, 'bands', 'uint32', nodeCount),
    predecessors: createTransientView(graph, 'predecessors', 'uint32', nodeCount),
    bandCounts: createTransientView(graph, 'bandCounts', 'uint32', 5),
    status: {
      converged: createTransientView(graph, 'converged', 'uint32', 1),
      iterationCount: createTransientView(graph, 'iterationCount', 'uint32', 1)
    }
  };
  const reachability = new GPUNetworkReachability({
    id,
    topology: roads,
    startNodes,
    activeIterations,
    costBreaks,
    maximumIterations: 256,
    localIterations,
    output
  });
  const {costs, bands, predecessors} = reachability.props.output;
  // #endregion example
  // Null devices cannot create compute pipelines, so structure-only builds skip compilation callbacks.
  const commandNodes = reachability.getCommandNodes(graph);
  graph.add(input.structureOnly ? commandNodes.map(withoutCompilation) : commandNodes);
  input.beforeCompile?.({output, commandNodes});
  const compiled = input.structureOnly ? graph.compile() : await graph.compileAsync();
  return {compiled, commandNodes, reachability, costs, bands, predecessors, output};
}

// Null devices support scheduling and resource allocation, but cannot create compute pipelines.
// Keep the original annotations and resources: only executable compilation callbacks are omitted.
function withoutCompilation(node: GPUCommandNode): GPUCommandNode {
  const {compile: _compile, compileAsync: _compileAsync, ...structure} = node;
  return {
    ...structure,
    compile: () => ({
      encode: () => {
        throw new Error('Structure-only graph cannot execute');
      }
    })
  };
}
`}},f={scenarios:[`gpu-spatial-analysis.kernel-density`],examples:{}},p={scenarios:[e.kernelDensity,e.localMoran,e.serviceAreas],examples:{recordRun:`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createGPUComputationRecord,
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  GPUSnapshotArena,
  GPUSnapshotTap
} from '@luma-contributors/analysis/computation-record';

// example:hide-start
declare const device: Device;
/** A graph that already holds an operation's kernels. */
declare const graph: GPUCommandGraph<void>;
/** The operation's output; its buffer was created with Buffer.COPY_SRC. */
declare const density: GraphDataView;
// example:hide-end
// 1. Add a copy node last. It reads the output into a buffer the CPU can map.
const arena = new GPUSnapshotArena(device, {
  id: 'run-snapshots',
  slots: [
    {name: 'density', byteLength: density.length * density.rowByteLength, elementType: 'f32'}
  ],
  capacity: 1
});
graph.add(new GPUSnapshotTap({id: 'run-snapshot-tap', arena, views: {density}}));

// 2. Compile once. The compiled graph publishes its order, buffers and hazards.
const compiled = graph.compile();

// 3. Describe it. Reads only the public id, stats and preflight report.
const description = describeGPUCommandGraph(compiled);

// 4. Assemble the record: the graph, the device and the domain.
const record = createGPUComputationRecord({
  id: 'kernel-density',
  adapter: getGPUComputationAdapter(device),
  domains: ['gpu-spatial-analysis'],
  graphs: [description]
});

// 5. Encode and submit once, then read the tap's copy.
const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());
const snapshots = await arena.read({
  graph: description.graph.id,
  resources: {density: 'density'}
});
const captures = createGPUComputationCaptures({
  executionId: record.executionId,
  snapshots,
  runId: record.runId
});
// example:hide-start
export {captures};
// example:hide-end
`}},m={scenarios:[e.kernelDensity],examples:{compileOnce:`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGPUKernelDensityParameterValues,
  GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
  GPUKernelDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// example:hide-start
import type {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

declare const device: Device;
declare const graph: GPUCommandGraph<void>;
declare const positions: GraphDataView<'float32x2'>;
declare const density: GraphDataView<'float32'>;
declare const grid: ReturnType<typeof makeGPURasterMetadata>;
declare const extent: readonly [number, number, number, number];
// example:hide-end
// Construct: props are validated here, once.
const parameters = new GPUParameterBuffer(device, {
  id: 'kde-parameters',
  format: 'float32',
  length: GPU_KERNEL_DENSITY_PARAMETER_LENGTH, // 9 float32 values
  values: getGPUKernelDensityParameterValues({bandwidth: 400}, grid)
});
graph.add(
  new GPUKernelDensity({
    positions,
    parameters: parameters.importToGraph(graph),
    grid,
    indexGridSize: [32, 32],
    indexBounds: extent,
    output: {values: density}
  })
);

// Compile, once: order the kernels, create the pipelines. The graph freezes.
const compiled = await graph.compileAsync();

// Every frame: record the fixed order and submit it. Nothing is re-planned.
function encodeFrame(): void {
  const commandEncoder = device.createCommandEncoder();
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());
}

// Between frames: a new bandwidth is a buffer write, not a recompile.
function setBandwidth(bandwidth: number): void {
  parameters.write(getGPUKernelDensityParameterValues({bandwidth}, grid));
}
// example:hide-start
export {encodeFrame, setBandwidth};
// example:hide-end
`}},h={scenarios:[],examples:{}},g={scenarios:[],examples:{describeGraph:`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describeGPUCommandGraph} from '@luma-contributors/analysis/computation-record';

// example:hide-start
/** Any graph, built by any code, compiled by published luma.gl. */
declare const compiled: CompiledGPUCommandGraph<void>;
// example:hide-end
// luma.gl publishes what the compiler decided: the order, and per node its buffers and usage.
const order = compiled.stats.nodeOrder;
const firstNode = compiled.preflight.nodes[0];
const firstNodeBuffers = (firstNode.resources ?? []).map(resource => resource.id);

// The description reads only those fields. No hook, no private field, no patched graph.
const {graph} = describeGPUCommandGraph(compiled);
const hazards = graph.edges.filter(edge => ['raw', 'war', 'waw'].includes(edge.kind));
const bufferBytes = (graph.resources ?? []).reduce((total, resource) => total + resource.bytes, 0);

// example:hide-start
export {order, firstNodeBuffers, hazards, bufferBytes};
// example:hide-end
`}},_=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import {
  registerRecipeContributor,
  isRecipeDeclaration,
  scheduleRecipeStage
} from './prepared-recipe';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  createTransientView,
  GPUReduction,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUCellAggregationFamily, GPUCellTable} from '../cell-aggregation/index';
import {GPUCellTopology, getCellTopologyStride} from '../cell-topology/index';
import type {GPURecipeStatusPort} from '../../analysis-core/contracts/index';
import {
  getGPUSpatialWeightsCapacityPlan,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput
} from '../spatial-weights/index';

/**
 * The only caller-ownership override categories accepted by recipes.
 *
 * \`inputs\` reuses a caller-owned value produced before the recipe, \`outputs\` publishes terminal
 * values, and \`scratch\` exposes storage that only connects recipe stages. Algorithm parameters
 * remain ordinary recipe properties because they are not ownership overrides.
 */
export type GPURecipeOverrides<
  Inputs extends object = Record<never, never>,
  Outputs extends object = Record<never, never>,
  Scratch extends object = Record<never, never>
> = {
  inputs?: Inputs;
  outputs?: Outputs;
  scratch?: Scratch;
};

/** Fields every recipe result carries. */
export type GPURecipeResult<
  Outputs extends Record<string, unknown> = Record<string, unknown>,
  Intermediates extends Record<string, unknown> = Record<string, unknown>
> = {
  /** Contributors the recipe added to the graph, in declaration (execution) order. */
  contributors: readonly GPUCommandNodeProducer<never>[];
  /** Stable named terminal outputs. */
  outputs: Outputs;
  /** Useful named ports between stages; graph-owned views are not readback-capable. */
  intermediates: Intermediates;
  /** GPU-resident completeness and convergence views, grouped by the responsible stage. */
  status: GPURecipeStatusPort;
};

/**
 * Collects contributors and schedules them in chain order. The deferred mode constructs and
 * expands every stage before \`addDeferredStages\` schedules the recipe's command nodes.
 * @internal
 */
export class RecipeBuilder<Parameters> {
  readonly contributors: GPUCommandNodeProducer<Parameters>[] = [];
  /** Names of the glue stages added with {@link RecipeBuilder.addGlue}, in declaration order. */
  readonly glueStages: {stage: string; nodes: readonly GPUCommandNode<Parameters>[]}[] = [];

  constructor(
    readonly graph: GPUCommandGraph<Parameters>,
    /** Hold stages until \`addDeferredStages\`, so every constructor can validate first. */
    private readonly deferGraphAddition = isRecipeDeclaration(graph)
  ) {}

  /** Records and returns \`contributor\`; schedules it now unless graph addition is deferred. */
  add<Contributor extends GPUCommandNodeProducer<Parameters>>(
    contributor: Contributor
  ): Contributor {
    registerRecipeContributor(this.graph, contributor);
    if (isRecipeDeclaration(this.graph))
      scheduleRecipeStage(this.graph, () => this.graph.add(contributor));
    else if (!this.deferGraphAddition) this.graph.add(contributor);
    this.contributors.push(contributor);
    return contributor;
  }

  /**
   * Adds raw glue nodes (adapter kernels that are not a contributor) under a stage name. The
   * nodes join the graph in call order (on flush in deferred mode), and appear in {@link RecipeBuilder.contributors} as
   * one producer, so a recipe result lists every stage it added, and in
   * {@link RecipeBuilder.glueStages} under \`stage\`.
   */
  addGlue(
    stage: string,
    nodes: GPUCommandNode<Parameters> | readonly GPUCommandNode<Parameters>[]
  ): void {
    const stageNodes = 'type' in nodes ? [nodes] : nodes;
    if (isRecipeDeclaration(this.graph))
      scheduleRecipeStage(this.graph, () => this.graph.add(stageNodes));
    else if (!this.deferGraphAddition) this.graph.add(stageNodes);
    this.glueStages.push({stage, nodes: stageNodes});
    this.contributors.push({getCommandNodes: () => stageNodes});
  }

  /** Records a published addToGraph stage or raw adapter as one ordered contributor. */
  addAdapter(addNodes: (graph: GPUCommandGraph<Parameters>) => void): void {
    this.add({getCommandNodes: graph => captureGraphCommandNodes(graph, () => addNodes(graph))});
  }

  /** Expands all deferred stages before scheduling any, preserving declaration order. */
  addDeferredStages(): void {
    // The enclosing recipe owns the ordered ledger, including interleaved raw stages.
    if (isRecipeDeclaration(this.graph)) return;
    validateRequirement(
      'RecipeBuilder',
      this.deferGraphAddition,
      'deferred graph addition must be enabled'
    );
    const nodes = this.contributors.flatMap(contributor => [
      ...contributor.getCommandNodes(this.graph)
    ]);
    this.graph.add(nodes);
  }
}

/** One count produced by {@link addRecipeCountNodes}. @internal */
export type RecipeCount = {
  /** Name of the count; the per-row value is bound as \`\${name}Values\` in the row body. */
  name: string;
  /** Caller-owned or graph-owned one-row uint32 view receiving the total. */
  output: GraphDataView<'uint32'>;
  /** WGSL \`u32\` expression for the row's contribution, evaluated once per invocation. */
  value: string;
};

/** Properties of {@link addRecipeCountNodes}. @internal */
export type RecipeCountNodesProps = {
  /** Stage name, also the node ID prefix. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** Number of rows (cells, groups, samples) the counts cover; one invocation per row. */
  rowCount: number;
  bindings: readonly WGSLKernelBinding[];
  declarations?: string;
  /** WGSL statements that run before the counts' \`value\` expressions, for shared locals. */
  prelude?: string;
  counts: readonly RecipeCount[];
};

/**
 * Adds a row-parallel kernel that writes one uint32 contribution per row for each count, then a
 * published \`GPUReduction\` sum per count. Replaces a single invocation looping over every row.
 * Integer sums are exact, so the totals equal the serial loop's. The kernel may also write other
 * per-row outputs from \`prelude\`.
 * @internal
 */
export function addRecipeCountNodes<Parameters>(
  builder: RecipeBuilder<Parameters>,
  props: RecipeCountNodesProps
): void {
  const {graph} = builder;
  // Same boundary checks as the recipe kernels: graph-owned, packed, outputs disjoint from inputs.
  validateGraphViewsBelongToGraph(
    props.id,
    graph,
    props.bindings.map(binding => binding.view)
  );
  for (const binding of props.bindings)
    validateSinglePackedView(binding.view, [binding.view.format], props.id);
  validateGraphOutputsDisjointFromInputs(
    props.id,
    props.bindings.filter(binding => binding.access !== 'read').map(binding => binding.view),
    props.bindings.filter(binding => binding.access === 'read').map(binding => binding.view)
  );
  const values = props.counts.map(count =>
    createTransientView(graph, \`\${props.id}-\${count.name}-values\`, 'uint32', props.rowCount)
  );
  builder.addGlue(props.id, [
    createWGSLKernelNode<Parameters>(graph, {
      id: props.id,
      operation: props.operation,
      variant: 'count-rows',
      bindings: [
        ...props.bindings,
        ...props.counts.map(
          (count, index) =>
            ({
              name: \`\${count.name}Values\`,
              view: values[index],
              type: 'u32',
              access: 'read_write'
            }) as const
        )
      ],
      invocationCount: props.rowCount,
      declarations: props.declarations,
      body: \`\${props.prelude ?? ''}
\${props.counts
  .map(count => \`\${count.name}Values[\${count.name}ValuesOffset + index] = \${count.value};\`)
  .join('\\n')}\`
    })
  ]);
  props.counts.forEach((count, index) => {
    builder.add(
      new GPUReduction({
        id: \`\${props.id}-\${count.name}-sum\`,
        input: values[index],
        output: count.output,
        operation: 'sum'
      })
    );
  });
}

/**
 * Returns \`provided\` when given, else a graph-owned transient view. Transient views carry data
 * between nodes but cannot be read back, so pass a caller-owned view for every result you need.
 * @internal
 */
export function getOrCreateView<Format extends VertexFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  format: Format,
  length: number,
  provided?: GraphDataView<Format>
): GraphDataView<Format> {
  return provided ?? createTransientView(graph, id, format, length);
}

/** Caller-provided parts of a recipe-allocated {@link GPUCellTable}. @internal */
export type ProvidedCellTable = Omit<Partial<GPUCellTable>, 'status'> & {
  status?: Partial<GPUCellTable['status']>;
};

/** Caller-provided parts of a recipe-allocated {@link GPUSpatialWeightsOutput}. @internal */
export type ProvidedSpatialWeightsOutput = Partial<GPUSpatialWeights> & {
  status?: Partial<GPUSpatialWeightsOutput['status']>;
};

/**
 * Creates a {@link GPUSpatialWeightsOutput} with \`rowCount + 1\` offsets, the slot capacity of
 * \`getGPUSpatialWeightsCapacityPlan({rowCount, capacity})\` (the same plan the caller's grow loop
 * reads back with the published \`requiredCount\`), an \`overflow\` flag and a \`requiredCount\`
 * scalar. Each view is \`provided\` when given, else graph-owned.
 * @internal
 */
export function createTransientSpatialWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rowCount: number,
  capacity: number,
  provided?: ProvidedSpatialWeightsOutput
): GPUSpatialWeightsOutput {
  const slots = getGPUSpatialWeightsCapacityPlan({rowCount, capacity}).capacity;
  return {
    offsets: getOrCreateView(graph, \`\${id}-offsets\`, 'uint32', rowCount + 1, provided?.offsets),
    neighbors: getOrCreateView(graph, \`\${id}-neighbors\`, 'uint32', slots, provided?.neighbors),
    weights: getOrCreateView(graph, \`\${id}-weights\`, 'float32', slots, provided?.weights),
    ...(provided?.distances ? {distances: provided.distances} : {}),
    status: {
      overflow: getOrCreateView(graph, \`\${id}-overflow\`, 'uint32', 1, provided?.status?.overflow),
      requiredCount: getOrCreateView(
        graph,
        \`\${id}-required-count\`,
        'uint32',
        1,
        provided?.status?.requiredCount
      ),
      ...(provided?.status?.candidateOverflow
        ? {candidateOverflow: provided.status.candidateOverflow}
        : {})
    }
  };
}

/**
 * Adds one node writing \`offsets[i] = i\` for \`i\` in \`[0, count]\` and returns the \`count + 1\`
 * view: the feature or polygon offsets of a layout with one part per row. With \`builder\`, the node
 * is registered as a named glue stage of the recipe.
 * @internal
 */
export function createIdentityOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: number,
  builder?: RecipeBuilder<Parameters>
): GraphDataView<'uint32'> {
  const offsets = createTransientView(graph, \`\${id}-offsets\`, 'uint32', count + 1);
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: \`\${id}-offsets\`,
    operation: 'GPURecipeIdentityOffsets',
    bindings: [{name: 'offsets', view: offsets, type: 'u32', access: 'read_write'}],
    invocationCount: count + 1,
    body: 'offsets[offsetsOffset + index] = index;'
  });
  if (builder) builder.addGlue(\`\${id}-identity-offsets\`, node);
  else graph.add(node);
  return offsets;
}

/**
 * Adapter: turns a cell table into the per-row analysis columns the statistics read.
 *
 * \`values[i]\` is \`sumValues[i]\` when the table has it, else the row count as f32. \`mask[i]\` is 1
 * for occupied rows and 0 for the empty tail of the capacity-bounded table.
 * @internal
 */
export function addCellTableColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  table: Pick<GPUCellTable, 'counts' | 'sumValues'>,
  output: {values: GraphDataView<'float32'>; mask: GraphDataView<'uint32'>}
): void {
  const rows = table.counts.length;
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: \`\${id}-cell-table-columns\`,
      operation: 'GPURecipeCellTableColumns',
      bindings: [
        {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
        ...(table.sumValues
          ? [{name: 'sumValues', view: table.sumValues, type: 'f32', access: 'read'} as const]
          : []),
        {name: 'columnValues', view: output.values, type: 'f32', access: 'read_write'},
        {name: 'columnMask', view: output.mask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      body: \`let rowCount = counts[countsOffset + index];
  columnValues[columnValuesOffset + index] = \${
    table.sumValues ? 'sumValues[sumValuesOffset + index]' : 'f32(rowCount)'
  };
  columnMask[columnMaskOffset + index] = select(0u, 1u, rowCount > 0u);\`
    })
  );
}

/** Throws a recipe-prefixed error when \`condition\` is false. @internal */
export function assertRecipe(id: string, condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(\`\${id} \${message}\`);
  }
}

/**
 * WGSL for a binary search of a 64-bit cell key in a sorted table of little-endian \`(low, high)\`
 * \`uint32x2\` rows, as written by \`GPUCellAggregation\`. Defines \`\${functionName}(key: vec2u) -> u32\`
 * with \`key = vec2u(high, low)\`, returning the row or \`0xffffffffu\` when the key is absent.
 * \`tableName\` must be a bound \`u32\` array (with its generated \`\${tableName}Offset\` constant) and
 * \`countExpression\` a WGSL \`u32\` expression for the number of occupied leading rows.
 * @internal
 */
export function getCellKeySearchWGSL(
  functionName: string,
  tableName: string,
  countExpression: string
): string {
  return \`fn \${functionName}(key: vec2u) -> u32 {
  let rowCount = \${countExpression};
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let middleLow = \${tableName}[\${tableName}Offset + 2u * middle];
    let middleHigh = \${tableName}[\${tableName}Offset + 2u * middle + 1u];
    if (middleHigh < key.x || (middleHigh == key.x && middleLow < key.y)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowCount && \${tableName}[\${tableName}Offset + 2u * low] == key.y && \${tableName}[\${tableName}Offset + 2u * low + 1u] == key.x) {
    return low;
  }
  return 0xffffffffu;
}\`;
}

/** Properties of {@link addCellDiskWeightsNodes}. @internal */
export type CellDiskWeightsProps = {
  family: GPUCellAggregationFamily;
  /** Sorted, unique study cell keys (\`uint32x2\` little-endian rows); occupied rows lead. */
  cells: GraphDataView<'uint32x2'>;
  /** One-row scalar with the number of occupied leading rows of \`cells\`. */
  count: GraphDataView<'uint32'>;
  /** Per-row mask for the topology: nonzero rows get a neighborhood. */
  mask: GraphDataView<'uint32'>;
  /** Grid distance of the disk, \`k >= 1\`. */
  rings: number;
  /** Output CSR (\`rows + 1\` offsets, \`neighborCapacity\` slots). */
  weights: GPUSpatialWeights;
  /** One-row scalar receiving 1 when \`neighborCapacity\` truncated the CSR. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped slot total, for the capacity grow loop. */
  requiredCount?: GraphDataView<'uint32'>;
};

/**
 * Builds a binary topological CSR over a sorted cell table: \`GPUCellTopology\` disk of radius
 * \`rings\`, a binary search of every disk key in the table, a count pass, an inclusive scan, and a
 * write pass that stores each row's neighbor indices (self excluded, ascending) with weight 1.
 * Neighbors absent from the table are dropped. When the total exceeds the slot capacity the
 * offsets are clamped to it and \`overflow\` is set. Returns the topology stride.
 * @internal
 */
export function addCellDiskWeightsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  builder: RecipeBuilder<Parameters>,
  id: string,
  props: CellDiskWeightsProps
): number {
  const {cells, weights} = props;
  const rows = cells.length;
  const capacity = weights.neighbors.length;
  const diskCells = createTransientView(
    graph,
    \`\${id}-disk-cells\`,
    'uint32x2',
    rows * getCellTopologyStride(props.family, {type: 'disk', k: props.rings})
  );
  const topology = builder.add(
    new GPUCellTopology({
      id: \`\${id}-disk\`,
      family: props.family,
      operation: {type: 'disk', k: props.rings},
      cells,
      mask: props.mask,
      output: {
        cells: diskCells
      }
    })
  );
  const stride = topology.stride;
  const counts = createTransientView(graph, \`\${id}-slot-counts\`, 'uint32', rows);
  const inclusive = createTransientView(graph, \`\${id}-slot-ends\`, 'uint32', rows);
  const declarations = \`const STRIDE: u32 = \${stride}u;
const ROWS: u32 = \${rows}u;
const CAPACITY: u32 = \${capacity}u;
const NOT_FOUND: u32 = 0xffffffffu;\`;
  const searchDeclarations = \`\${declarations}
\${getCellKeySearchWGSL('findStudyCell', 'cells', 'countScalar[countScalarOffset]')}\`;
  // Reads disk entry \`entry\` of this row as vec2u(high, low); zero entries are padding.
  const readEntry = \`let entryLow = disk[diskOffset + 2u * (index * STRIDE + entry)];
    let entryHigh = disk[diskOffset + 2u * (index * STRIDE + entry) + 1u];
    if (entryLow == 0u && entryHigh == 0u) {
      continue;
    }
    let found = findStudyCell(vec2u(entryHigh, entryLow));
    if (found == NOT_FOUND || found == index) {
      continue;
    }\`;
  builder.addGlue(
    'slot-counts',
    createWGSLKernelNode<Parameters>(graph, {
      id: \`\${id}-slot-counts\`,
      operation: 'GPUCellDiskWeights',
      variant: 'count',
      bindings: [
        {name: 'disk', view: diskCells, type: 'u32', access: 'read'},
        {name: 'cells', view: cells, type: 'u32', access: 'read'},
        {name: 'countScalar', view: props.count, type: 'u32', access: 'read'},
        {name: 'slotCounts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: searchDeclarations,
      body: \`var count = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    \${readEntry}
    count++;
  }
  slotCounts[slotCountsOffset + index] = count;\`
    })
  );
  builder.add(
    new GPUScan({id: \`\${id}-slot-scan\`, input: counts, output: inclusive, mode: 'inclusive'})
  );
  builder.addGlue(
    'offsets',
    createWGSLKernelNode<Parameters>(graph, {
      id: \`\${id}-offsets\`,
      operation: 'GPUCellDiskWeights',
      variant: 'offsets',
      bindings: [
        {name: 'slotEnds', view: inclusive, type: 'u32', access: 'read'},
        {name: 'rowOffsets', view: weights.offsets, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
        ...(props.requiredCount
          ? [
              {
                name: 'requiredCount',
                view: props.requiredCount,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: rows + 1,
      declarations,
      body: \`var rowOffset = 0u;
  if (index > 0u) {
    rowOffset = min(slotEnds[slotEndsOffset + index - 1u], CAPACITY);
  }
  rowOffsets[rowOffsetsOffset + index] = rowOffset;
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, slotEnds[slotEndsOffset + ROWS - 1u] > CAPACITY);
    \${props.requiredCount ? 'requiredCount[requiredCountOffset] = slotEnds[slotEndsOffset + ROWS - 1u];' : ''}
  }\`
    })
  );
  builder.addGlue(
    'write',
    createWGSLKernelNode<Parameters>(graph, {
      id: \`\${id}-write\`,
      operation: 'GPUCellDiskWeights',
      variant: 'write',
      bindings: [
        {name: 'disk', view: diskCells, type: 'u32', access: 'read'},
        {name: 'cells', view: cells, type: 'u32', access: 'read'},
        {name: 'countScalar', view: props.count, type: 'u32', access: 'read'},
        {name: 'slotEnds', view: inclusive, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
        {name: 'weightValues', view: weights.weights, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: searchDeclarations,
      body: \`var start = 0u;
  if (index > 0u) {
    start = slotEnds[slotEndsOffset + index - 1u];
  }
  var written = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    \${readEntry}
    let slot = start + written;
    if (slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = found;
      weightValues[weightValuesOffset + slot] = 1.0;
    }
    written++;
  }
  // Ascending neighbor index within the row (insertion sort of the slots that fit).
  let stored = select(0u, min(written, CAPACITY - start), start < CAPACITY);
  for (var i = 1u; i < stored; i++) {
    let value = neighbors[neighborsOffset + start + i];
    var j = i;
    loop {
      if (j == 0u || neighbors[neighborsOffset + start + j - 1u] <= value) {
        break;
      }
      neighbors[neighborsOffset + start + j] = neighbors[neighborsOffset + start + j - 1u];
      j--;
    }
    neighbors[neighborsOffset + start + j] = value;
  }\`
    })
  );
  return stride;
}

/** Exact output sizes keep recipe indexing independent of caller buffer capacity. @internal */
export function getRecipeView<Format extends VertexFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  format: Format,
  length: number,
  provided?: GraphDataView<Format>
): GraphDataView<Format> {
  if (provided) {
    validateSinglePackedView(provided, [format], id);
    // The provided view must have the exact row count promised by the recipe output.
    validateViewLength(id, 'provided output', provided, length);
    validateGraphViewsBelongToGraph(id, graph, [provided]);
  }
  return getOrCreateView(graph, id, format, length, provided);
}
/** Recipe arithmetic follows the same ownership and packed-layout contract as contributors. @internal */
export function createRecipeKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
) {
  validateGraphViewsBelongToGraph(
    props.id,
    graph,
    props.bindings.map(binding => binding.view)
  );
  for (const binding of props.bindings)
    validateSinglePackedView(binding.view, [binding.view.format], props.id);
  validateGraphOutputsDisjointFromInputs(
    props.id,
    props.bindings.filter(binding => binding.access !== 'read').map(binding => binding.view),
    props.bindings.filter(binding => binding.access === 'read').map(binding => binding.view)
  );
  return createWGSLKernelNode(graph, props);
}

/** Validate the whole recipe boundary, including buffers used by nonadjacent stages. @internal */
export function validateRecipeBoundary<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly (GraphDataView | undefined)[],
  outputs: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, [...inputs, ...outputs]);
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  const buffers = outputs
    .filter(view => view !== undefined)
    .map(view =>
      view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer
    );
  // Independent named outputs must not share a buffer, even through disjoint views.
  validateRequirement(
    id,
    new Set(buffers).size === buffers.length,
    'outputs must not share a buffer'
  );
}
`,v=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUStatusScalar
} from '../kernels/gpu-contributor-types';

// The scalar, cardinality and overflow ports sit in utils so the compact-output contract there
// is typed with them; this module is their public home.
export type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUStatusScalar
} from '../kernels/gpu-contributor-types';

/** Completeness of an intermediate candidate stage. */
export type GPUCandidateStatusPort = {
  /** One when refinement may be incomplete because an intermediate capacity was too small. */
  candidateOverflow?: GPUStatusScalar;
};

/** Numerical classifications that could not be certified under the selected precision policy. */
export type GPUUncertaintyStatusPort = {
  /** Exact number of classifications resolved through the documented uncertainty fallback. */
  uncertainCount?: GPUStatusScalar;
};

/** Lon/lat polygon rings a degree-plane consumer could not interpret (results may be wrong there). */
export type GPULonLatPolygonStatusPort = {
  /** Rings with an edge whose raw longitude step exceeds 180 (antimeridian crossing, not split). */
  antimeridianUncertainCount?: GPUStatusScalar;
  /** Rings whose wrapped longitude steps sum to +-360 (enclose a pole, not capped). */
  poleUncertainCount?: GPUStatusScalar;
};

/** Input validity diagnostics. */
export type GPUInvalidStatusPort = {
  /**
   * Number of input rows rejected by validation or, for \`GPURasterFillNoData\`, the number of
   * cells of the result that are still invalid after the fill.
   */
  invalidCount?: GPUStatusScalar;
};

/**
 * Status for bounded iterative work: the one convergence-status shape. Solvers
 * (\`GPUSolverStatusPort\`), searches and iterative contributors use
 * these field names for the executed iteration count and the stopping flags.
 */
export type GPUIterationStatusPort = {
  /** Number of iterations, rounds or passes actually executed. */
  iterationCount?: GPUStatusScalar;
  /** One when the operation met its stopping condition. */
  converged?: GPUStatusScalar;
  /** One when the iteration budget was exhausted before convergence. */
  iterationLimitReached?: GPUStatusScalar;
};

/** Status for algorithms that intentionally return an approximation. */
export type GPUApproximationStatusPort = {
  /** One when the result uses an intentional approximation rather than exact semantics. */
  approximation?: GPUStatusScalar;
};

/** Composable status vocabulary for a capacity-bounded analytical result. */
export type GPUBoundedResultStatusPort = GPUCardinalityPort &
  GPUOverflowStatusPort &
  GPUCandidateStatusPort &
  GPUUncertaintyStatusPort &
  GPUInvalidStatusPort &
  GPUIterationStatusPort &
  GPUApproximationStatusPort;

/**
 * Vertex sub-count of generated geometry (contributor contract rule 8). \`count\` and
 * \`requiredCount\` of generated geometry count features; the vertex totals have these names.
 */
export type GPUVertexCountStatusPort = {
  /** Number of vertices written, clamped to the vertex capacity. */
  vertexCount?: GPUStatusScalar;
  /** Unclamped number of vertices the result needs. */
  requiredVertexCount?: GPUStatusScalar;
};

/** Ring sub-count of generated polygon geometry (contributor contract rule 8). */
export type GPURingCountStatusPort = {
  /** Number of rings written, clamped to the ring capacity. */
  ringCount?: GPUStatusScalar;
  /** Unclamped number of rings the result needs. */
  requiredRingCount?: GPUStatusScalar;
};

/**
 * Status of generated geometry (contributor contract rule 8): \`count\` and \`requiredCount\` count
 * output features, \`overflow\` is one when any final capacity (features, rings or vertices) dropped
 * rows, and the sub-levels have their own named counts.
 */
export type GPUGeneratedGeometryStatusPort = GPUBoundedResultStatusPort &
  GPUVertexCountStatusPort &
  GPURingCountStatusPort;

/** Status attributed to one named recipe stage. */
export type GPURecipeStageStatusPort = {
  stage: string;
  status: Partial<GPUBoundedResultStatusPort>;
};

/** Aggregate status that preserves the stage responsible for each condition. */
export type GPURecipeStatusPort = {
  stages: readonly GPURecipeStageStatusPort[];
};

/** Validates the format and scalar shape of every present status field. */
export function validateGPUStatusPort(
  id: string,
  status: Partial<GPUBoundedResultStatusPort> & GPULonLatPolygonStatusPort
): void {
  for (const [name, scalar] of Object.entries(status)) {
    if (!scalar) {
      continue;
    }
    if (scalar.format !== 'uint32' || scalar.length < 1) {
      // Status values are GPU-written uint32 scalars.
      throw new Error(\`\${id} \${name} must contain one uint32 row\`);
    }
  }
}

/** Validates every stage of a recipe status aggregate. */
export function validateGPURecipeStatusPort(id: string, status: GPURecipeStatusPort): void {
  const stages = new Set<string>();
  for (const stage of status.stages) {
    if (!stage.stage || stages.has(stage.stage)) {
      // Stage names identify the operation responsible for an incomplete result.
      throw new Error(\`\${id} recipe status stages must have unique non-empty names\`);
    }
    stages.add(stage.stage);
    validateGPUStatusPort(\`\${id} stage \${stage.stage}\`, stage.status);
  }
}
`,y=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addDriveTimeCatchmentRecipe,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs: roads, facility and demand longitude/latitude, and ascending drive-time breaks. */
export type DriveTimeCatchmentRecipeExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  facilityPositions: GraphDataView<'float32x2'>;
  demandPositions: GraphDataView<'float32x2'>;
  costBreaks: GraphDataView<'float32'>;
};

/** Adds the recipe shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: DriveTimeCatchmentRecipeExampleInput) {
  const {graph, roads, facilityPositions, demandPositions, costBreaks} = input;
  // #region example
  const catchment = addDriveTimeCatchmentRecipe(graph, {
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    topology: roads,
    facilities: facilityPositions,
    demand: demandPositions,
    costBreaks,
    maximumIterations: 24
  });
  const {demandTimes, demandBands, bands} = catchment;
  // #endregion example
  return {catchment, demandTimes, demandBands, bands};
}
`,b=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  addHotSpotAnalysisRecipe,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

/** Inputs the lens scenario uploads from the Chicago North Side observations and areas. */
export type HotSpotAnalysisRecipeExampleInput = {
  /** iNaturalist observations as longitude/latitude degrees. */
  observations: GraphDataView<'float32x2'>;
  /** Study-area polygons (community areas) as longitude/latitude four-level views. */
  studyArea: {
    positions: GraphDataView<'float32x2'>;
    featureOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringOffsets: GraphDataView<'uint32'>;
  };
  /** Rows of the study table: an upper bound on the Quadbin cells inside the study area. */
  cellCapacity: number;
  /** Candidate cells the cover tests: an upper bound from the polygon bounding boxes. */
  candidateCapacity: number;
};

/** Getis-Ord Gi* hot and cold spots of observation density on Quadbin cells inside the study area. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, studyArea, cellCapacity, candidateCapacity}: HotSpotAnalysisRecipeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const zScores = output('zScores', 'float32', cellCapacity);
  const pValues = output('pValues', 'float32', cellCapacity);
  const bins = createTransientView(graph, 'bins', 'sint32', cellCapacity, usage);
  const neighborCounts = output('neighborCounts', 'uint32', cellCapacity);
  const pointsOutsideStudyArea = output('pointsOutsideStudyArea', 'uint32', 1);
  const centers = output('centers', 'float32x2', cellCapacity);
  const values = output('values', 'float32', cellCapacity);
  // #region example
  const hotSpots = addHotSpotAnalysisRecipe(graph, {
    source: {
      kind: 'points',
      positions: observations,
      family: 'quadbin',
      resolution: 16,
      studyArea: {
        kind: 'polygons',
        ...studyArea,
        candidateCapacity,
        coverCapacity: 2 * cellCapacity
      },
      cellCapacity
    },
    parameters,
    settings: {significanceLevel: 0.05},
    outputs: {zScores, pValues, bins, neighborCounts, status: {pointsOutsideStudyArea}},
    scratch: {centers, values}
  });
  // zScores at centers: bins +-1, 2, 3 are 90, 95, 99% confidence hot or cold spots
  // #endregion example
  return {
    hotSpots,
    zScores,
    pValues,
    bins,
    neighborCounts,
    pointsOutsideStudyArea,
    centers,
    values,
    parameters
  };
}
`,x=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import {
  addRateClusterMapRecipe,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPURateClusterMapRecipeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

/** Inputs the lens scenario uploads from the Chicago North Side tracts. */
export type RateClusterMapRecipeExampleInput = {
  /** Tract polygons as four-level views. */
  tracts: GPURateClusterMapRecipeProps['polygons'];
  /** Bird observations per tract. */
  birdObservations: GraphDataView<'uint32'>;
  /** All observations per tract, the population at risk of being a bird. */
  natureObservations: GraphDataView<'uint32'>;
};

/** Clusters of high and low bird shares among observations, from smoothed rates and queen weights. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, birdObservations, natureObservations}: RateClusterMapRecipeExampleInput
) {
  const count = birdObservations.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const permutationParameters = new GPUParameterBuffer(graph.device, {
    id: 'permutation-parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH
  });
  const palette = new GPUParameterBuffer(graph.device, {
    id: 'quadrant-palette',
    format: 'uint32',
    length: 5,
    values: Uint32Array.of(
      packGPUColor(200, 200, 200),
      packGPUColor(215, 25, 28),
      packGPUColor(171, 217, 233),
      packGPUColor(44, 123, 182),
      packGPUColor(253, 174, 97)
    )
  });
  const standardizedRates = output('standardizedRates', 'float32', count);
  const zScores = output('zScores', 'float32', count);
  const quadrants = output('quadrants', 'uint32', count);
  const colors = output('colors', 'uint32', count);
  const permutation = {
    pseudoPValues: output('pseudoPValues', 'float32', count),
    adjustedPValues: output('adjustedPValues', 'float32', count),
    significant: output('significant', 'uint32', count)
  };
  // #region example
  const clusters = addRateClusterMapRecipe(graph, {
    events: birdObservations,
    populations: natureObservations,
    polygons: tracts,
    criterion: 'queen',
    neighborCapacity: 4096,
    parameters,
    settings: {significanceLevel: 0.05, seed: 7, permutationCount: 199},
    permutation: {parameters: permutationParameters, maximumPermutations: 199},
    palette: palette.importToGraph(graph),
    outputs: {standardizedRates, zScores, quadrants, colors, permutation}
  });
  // quadrants colours each tract: 0 not significant, then high-high, low-high, low-low, high-low
  // #endregion example
  return {
    clusters,
    standardizedRates,
    zScores,
    quadrants,
    colors,
    permutation,
    parameters: [parameters, permutationParameters, palette]
  };
}
`,S=`// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  addSpaceTimeHotSpotsRecipe,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';

/** Slices of the cube: one per month of 2023. */
export const SPACE_TIME_SLICE_COUNT = 12;

/** Inputs the lens scenario uploads from \`s2-chicago-north-nature.json\`. */
export type SpaceTimeHotSpotsRecipeExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Observation times as little-endian Int64 epoch milliseconds. */
  timestamps: GraphDataView<'uint32x2'>;
  /** \`getGPUCalendarBucketsParameterValues(0)\`: UTC. */
  calendarParameters: GraphDataView<'sint32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Lattice columns. */
  width: number;
  /** Lattice rows. */
  height: number;
  /** Covered extent \`[minX, minY, maxX, maxY]\`, a whole number of cells. */
  bounds: readonly [number, number, number, number];
};

/** Timestamped observations to emerging hot spots on an 800 m lattice and 12 monthly slices. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    observations,
    timestamps,
    calendarParameters,
    spatialContext,
    width,
    height,
    bounds
  }: SpaceTimeHotSpotsRecipeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const cellCount = width * height;
  const binCount = cellCount * SPACE_TIME_SLICE_COUNT;
  const create = <Format extends 'float32' | 'uint32' | 'sint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const outputs = {
    cube: create('cube', 'uint32', binCount),
    giZScores: create('giZScores', 'float32', binCount),
    trendZ: create('trendZ', 'float32', cellCount),
    category: create('category', 'uint32', cellCount),
    hotSliceCount: create('hotSliceCount', 'uint32', cellCount),
    coldSliceCount: create('coldSliceCount', 'uint32', cellCount)
  };
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'emerging-hot-spot-parameters',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  });
  // #region example
  addSpaceTimeHotSpotsRecipe(graph, {
    timestamps,
    calendarParameters,
    slices: {step: 'month', origin: {year: 2023, month: 1}, count: 12},
    cells: {kind: 'lattice', positions: observations, spatialContext, width, height, bounds},
    settings: {radius: 1.5, temporalWindow: 1},
    parameters,
    outputs
  });
  // category per cell: the 17 ArcGIS emerging hot spot classes from the month-by-cell cube
  // #endregion example
  return {...outputs, parameters};
}
`,C=`// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// example:hide-start
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';

declare const graph: GPUCommandGraph<void>;
/** The elevation raster's grid and its values, already a view in \`graph\`. */
declare const grid: GPURasterMetadata;
declare const elevation: GraphDataView<'float32'>;
// example:hide-end
// A cell is a stream where at least 200 cells drain through it.
const values = getGPUTerrainFlowParameterValues({streamThreshold: 200}, grid);
const flowParameters = new GPUParameterBuffer(graph.device, {
  id: 'flow-parameters',
  format: 'float32',
  length: values.length,
  values
});
const drainage = addDrainageNetworkRecipe(graph, {
  flow: {
    grid,
    elevation: makeGPURasterBand(elevation, grid),
    parameters: flowParameters.importToGraph(graph),
    fillDepressions: true,
    resolveFlats: true,
    accumulationUnits: 'cells'
  },
  maximumSteps: grid.width * grid.height
});
const {streamOrder, catchments, network} = drainage.outputs;
// example:hide-start
export {catchments, network, streamOrder};
// example:hide-end
`;function w(e,t){let n=e.split(`
`),r=n.findIndex(e=>e.startsWith(t));if(r<0)return`// ${t} not found`;let i=r;for(;i>0&&/^\s*(\/\*\*|\*)/.test(n[i-1]);)i--;let a=n.findIndex((e,t)=>t>r&&e===`};`);return n.slice(i,a<0?r+1:a+1).join(`
`)}var T={scenarios:[],examples:{"local-moran":i,"hot-spots":b,"rate-clusters":x,"space-time":S,"drive-time":y,drainage:C,"recipe-result":[w(_,`export type GPURecipeResult<`),w(v,`export type GPURecipeStatusPort =`),w(v,`export type GPURecipeStageStatusPort =`)].join(`

`)}},E=[`anatomy`,`data-on-the-gpu`,`command-graph`,`scheduling`,`kernels`,`building-blocks`,`iteration`,`contract`,`workflows`,`lenses`,`this-page`,`upstream-first`,`glossary`],D=[{id:`run`,title:`One run`,lead:`Follow a single call from buffers to a picture.`},{id:`parallel`,title:`Parallel analysis`,lead:`How spatial algorithms become kernels that never loop over every row.`},{id:`practice`,title:`In practice`,lead:`Operations chained into a whole analysis.`},{id:`system`,title:`The system`,lead:`Records, packages and vocabulary.`}],O=[{slug:`anatomy`,part:`run`,title:`Anatomy of a run`,lead:`One call becomes a graph of GPU kernels. Follow kernel density from the code to the raster.`,...t},{slug:`data-on-the-gpu`,part:`run`,title:`Data on the GPU`,lead:`Every input and output is a typed view of a buffer. Offsets give flat arrays their shape.`,...l},{slug:`command-graph`,part:`run`,title:`The command graph`,lead:`Kernels declare what they read and write. The compiler orders them from that alone.`,...s},{slug:`scheduling`,part:`run`,title:`Scheduling`,lead:`Compile fixes the order once. Each frame only encodes it, and one small buffer changes the answer.`,...m},{slug:`kernels`,part:`run`,title:`Kernels`,lead:`One shader, thousands of invocations: how a dispatch covers the data.`,...f},{slug:`building-blocks`,part:`parallel`,title:`Parallel building blocks`,lead:`No kernel loops over every row. Scan, scatter, sort and a grid index do the work a loop would.`,...r},{slug:`iteration`,part:`parallel`,title:`Loops without readback`,lead:`Rounds are laid out in advance and gated on the GPU. A finished search runs zero workgroups.`,...d},{slug:`contract`,part:`parallel`,title:`The contributor contract`,lead:`Every operation follows the same rules, so every operation composes and fails the same way.`,...c},{slug:`workflows`,part:`practice`,title:`Workflows`,lead:`A real analysis chains operations in one graph. Five of them, step by step.`,...T},{slug:`lenses`,part:`system`,title:`Lenses`,lead:`Function pages show a run recorded on a real GPU. This is what a record holds.`,...p},{slug:`this-page`,part:`system`,title:`This page`,lead:`The canvas is drawn by luma.gl right now; these are its real numbers.`,...h},{slug:`upstream-first`,part:`system`,title:`Upstream first`,lead:`Gridlight adds no GPU runtime; it stacks on published luma.gl.`,...g},{slug:`glossary`,part:`system`,title:`Glossary`,lead:`Every term on these pages, who defines it and where it is explained.`,...u}];if(O.length!==E.length||O.some((e,t)=>e.slug!==E[t]))throw Error(`topics.ts and topic-slugs.ts list different pages`);function k(e){return O.find(t=>t.slug===e)}function A(e){return D.find(t=>t.id===e)??D[0]}function j(e){let t=O.findIndex(t=>t.slug===e);return{previous:O[t-1],next:O[t+1]}}export{A as a,e as c,j as i,D as n,o,k as r,n as s,O as t};