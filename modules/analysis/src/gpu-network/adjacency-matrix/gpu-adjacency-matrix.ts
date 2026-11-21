// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

import {validateNetworkViewBindings, validateNetworkWritableViews} from '../network-view-utils';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import {GPURasterBufferToTexture} from '@luma.gl/graph/gpu-raster';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPUNetworkAdjacency} from '../../analysis-core/contracts/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of words of the per-frame `parameters` view: `[rowStart, rowEnd, colStart, colEnd]`. */
export const GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH = 4;

/** Default fixed-point scale of weight sums: one unit of weight is 1024 integer steps. */
export const GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE = 1024;

/** Per-frame zoom window of one {@link GPUAdjacencyMatrix}, in matrix positions. Ends are exclusive. */
export type GPUAdjacencyMatrixSettings = {
  /** First row position. */
  rowStart: number;
  /** One past the last row position. */
  rowEnd: number;
  /** First column position. */
  colStart: number;
  /** One past the last column position. */
  colEnd: number;
  /** Matrix resolution `R` of the {@link GPUAdjacencyMatrix}, used to bound the window extents. */
  resolution: number;
};

/** Shader weight scales must remain positive, representable float32 literals. */
const ADJACENCY_MATRIX_MINIMUM_WEIGHT_SCALE = 2 ** -149;
/** Largest finite float32 scale; larger values generate an invalid infinite WGSL literal. */
const ADJACENCY_MATRIX_MAXIMUM_WEIGHT_SCALE = (2 - 2 ** -23) * 2 ** 127;

/**
 * Packs the per-frame zoom window into the four `uint32` words of the `parameters` view.
 *
 * Write the result with `GPUParameterBuffer.write`. An empty window (`end <= start`)
 * produces an all-zero matrix. The GPU computes `(position - start) * resolution` in `u32`, so
 * each extent times `settings.resolution` must stay below 2^32.
 *
 * @throws If a bound is not a uint32 integer or an extent times `resolution` reaches 2^32.
 */
export function getGPUAdjacencyMatrixParameterValues(
  settings: GPUAdjacencyMatrixSettings
): Uint32Array {
  const {rowStart, rowEnd, colStart, colEnd, resolution} = settings;
  for (const value of [rowStart, rowEnd, colStart, colEnd]) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window bounds must be uint32 integers');
    }
  }
  for (const extent of [rowEnd - rowStart, colEnd - colStart]) {
    if (extent * resolution > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window extent times resolution must be below 2^32');
    }
  }
  return new Uint32Array([rowStart, rowEnd, colStart, colEnd]);
}

/**
 * Fixed-point weight contributed by one edge, exactly as the GPU kernel computes it: the float32
 * product `weight * scale`, rounded half up, clamped to `[0, 2^32 - 1]` (a product of 2^32 - 256
 * or more gives `2^32 - 1`). With a power-of-two scale the product is exact; with another scale
 * the product is rounded to float32 before the half-up rounding.
 */
export function getGPUAdjacencyMatrixFixedWeight(weight: number, scale: number): number {
  const scaled = Math.fround(Math.fround(weight) * Math.fround(scale));
  if (!(scaled > 0)) {
    return 0;
  }
  if (scaled >= 4294967040) {
    return 0xffffffff;
  }
  const whole = Math.floor(scaled);
  return whole + (scaled >= whole + 0.5 ? 1 : 0);
}

/**
 * Properties for {@link GPUAdjacencyMatrix}.
 *
 * Compile-time: node count, `resolution`, `directed`, `mirrorSlots`, `weightScale`, and
 * which optional views exist. Per-frame: CSR contents, masks, `order` and `parameters`.
 */
export type GPUAdjacencyMatrixProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'adjacency-matrix'`. */
  id?: string;
  /**
   * Forward CSR network. Neighbor indices `>= nodeCount` are dead slots. `weights` are required
   * for weight outputs. Every row offset must lie within the stored neighbor capacity. For
   * `GPUNetworkSubgraphFilter.output.inducedTopology`, first submit the filter and verify its
   * `status.overflow` is zero before binding this topology: on overflow its offsets describe
   * required slots beyond the stored neighbors. Recheck after every filter submission.
   */
  topology: GPUNetworkAdjacency;
  /** Treat slots as directed edges (row = source, column = target). Default false. */
  directed?: boolean;
  /**
   * Undirected only, default false. An undirected CSR lists both directions (as in
   * `GPUNetworkStatistics` and gpu-graph), so by default each slot is written once and the matrix
   * is symmetric because the CSR is. When true, each slot `(u, v)` is also written transposed at
   * `(v, u)` (a self-loop once), for edge-list-style CSRs that list each edge once.
   */
  mirrorSlots?: boolean;
  /** Optional per-node mask, nonzero = live. */
  nodeMask?: GraphDataView<'uint32'>;
  /** Optional per-slot mask, nonzero = live. A slot is live iff both endpoints are live too. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Optional permutation view (node to matrix position, `nodeCount` rows). Default identity. */
  order?: GraphDataView<'uint32'>;
  /**
   * Per-frame `'uint32'` view of four words, see {@link getGPUAdjacencyMatrixParameterValues}.
   * When omitted the window is `[0, nodeCount)` on both axes.
   */
  parameters?: GraphDataView<'uint32'>;
  /** Matrix resolution `R` in bins per axis, compile-time, 1 to 4096. */
  resolution: number;
  /**
   * Fixed-point scale for weight sums, default 1024. Per-edge weights are rounded to
   * `round(weight * scale)`, negative and NaN weights count 0, and sums wrap modulo 2^32.
   */
  weightScale?: number;
  /** Caller-owned outputs. */
  output: {
    /** `resolution * resolution` row-major `uint32` edge counts (`row * R + column`). */
    counts: GraphDataView<'uint32'>;
    /** Optional same-size fixed-point weight sums. Requires `weights`. */
    weightSums?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the maximum of `counts`. */
    maximumCount?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the maximum fixed-point weight sum; public weightSums are optional. */
    maximumWeightSum?: GraphDataView<'uint32'>;
    /** Optional `r32float` `R x R` storage texture; row = y, column = x. */
    texture?: GraphTextureView<'r32float'>;
    /** Texture contents: raw counts (default) or weight sums / weightScale. Weighted bins are transient if weightSums is omitted. */
    textureStatistic?: 'count' | 'weightSum';
  };
};

/**
 * Bins a graph into an `R x R` adjacency-matrix image on the GPU, for a matrix view linked to a
 * node-link view.
 *
 * A live slot `u -> v` lands in cell `(bin(position(u), rows), bin(position(v), columns))`, where
 * `position` is the optional `order` permutation (identity by default) and, for window
 * `[start, end)`, `bin(p) = floor((p - start) * R / (end - start))` for `start <= p < end`;
 * positions outside the window are dropped. Panning or zooming rewrites the four `parameters` words and
 * the next encoding rewrites every bin, with no recompilation.
 *
 * Counts are exact `u32` atomic adds. Weight sums are fixed-point `u32` atomic adds of
 * `round(weight * weightScale)` (the float32 product, rounded half up, which
 * `getGPUAdjacencyMatrixFixedWeight` reproduces exactly); integer addition is associative, so results are bit-identical
 * regardless of thread order. The cost is a quantization of at most half a step per edge and
 * wraparound at 2^32 total (4,194,304 units of weight per cell at the default scale). Float
 * atomics are not used. Dead nodes keep their matrix positions but write no cells. Maxima are
 * `atomicMax` over the finished bins. `R * R` zeroing, binning and the maxima are separate nodes.
 *
 * The binning kernels use one invocation per node that walks its row, like
 * `GPUNetworkStatistics`; a hub node serializes its own row. Node positions must be a
 * permutation if the caller wants distinct rows per node; this is not validated on the GPU.
 */
export class GPUAdjacencyMatrix implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAdjacencyMatrixProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  /** Bins per axis. */
  readonly resolution: number;
  /** Fixed-point scale of weight sums. */
  readonly weightScale: number;

  constructor(props: GPUAdjacencyMatrixProps) {
    this.id = props.id ?? 'adjacency-matrix';
    this.props = props;
    validateNetworkWritableViews(this.id, props.output, [
      ...(getFusedBinBindingCount(props) <=
      (props.topology.offsets.buffer.graph as GPUCommandGraph).device.limits
        .maxStorageBuffersPerShaderStage
        ? [{outputs: {counts: props.output.counts, weightSums: props.output.weightSums}}]
        : []),
      {
        outputs: {
          maximumCount: props.output.maximumCount,
          maximumWeightSum: props.output.maximumWeightSum
        },
        inputs: {
          counts: props.output.maximumCount ? props.output.counts : undefined,
          weightSums: props.output.maximumWeightSum ? props.output.weightSums : undefined
        }
      }
    ]);
    const {id} = this;
    this.nodeCount = validateGPUNetworkTopology(id, props.topology).nodeCount;
    this.resolution = props.resolution;
    validateCapacity(id, 'resolution', props.resolution, 1, 4096);
    if (this.nodeCount * props.resolution > 0xffffffff) {
      throw new Error(`${id} node count times resolution must be below 2^32`);
    }
    this.weightScale = props.weightScale ?? GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE;
    if (!Number.isFinite(this.weightScale) || this.weightScale <= 0) {
      throw new Error(`${id} weightScale must be positive and finite`);
    }
    validateFiniteNumber(
      id,
      'weightScale',
      this.weightScale,
      ADJACENCY_MATRIX_MINIMUM_WEIGHT_SCALE,
      ADJACENCY_MATRIX_MAXIMUM_WEIGHT_SCALE
    );
    for (const [name, view, length] of [
      ['nodeMask', props.nodeMask, this.nodeCount],
      ['order', props.order, this.nodeCount],
      ['edgeMask', props.edgeMask, props.topology.neighbors.length]
    ] as const) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain ${length} rows`);
      }
    }
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
      if (props.parameters.length < GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH) {
        throw new Error(`${id} parameters must contain at least four uint32 rows`);
      }
    }
    const {output} = props;
    validateRequirement(
      id,
      output.textureStatistic === undefined ||
        output.textureStatistic === 'count' ||
        output.textureStatistic === 'weightSum',
      `output.textureStatistic must be 'count' or 'weightSum'; received ${output.textureStatistic}`
    );
    const cellCount = props.resolution * props.resolution;
    for (const [name, view, length] of [
      ['counts', output.counts, cellCount],
      ['weightSums', output.weightSums, cellCount],
      ['maximumCount', output.maximumCount, 1],
      ['maximumWeightSum', output.maximumWeightSum, 1]
    ] as const) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
      if (
        name === 'maximumCount' || name === 'maximumWeightSum'
          ? view.length < 1
          : view.length !== length
      ) {
        throw new Error(
          `${id} output.${name} must contain ${length === 1 ? 'at least one row' : `exactly ${length} rows`}`
        );
      }
    }
    if (
      (output.weightSums || output.maximumWeightSum || output.textureStatistic === 'weightSum') &&
      !props.topology.weights
    ) {
      throw new Error(`${id} weight outputs require weights`);
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== props.resolution ||
        output.texture.height !== props.resolution)
    ) {
      throw new Error(`${id} output.texture must be r32float with resolution x resolution extent`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.counts, output.weightSums, output.maximumCount, output.maximumWeightSum],
      [
        ...getGPUNetworkTopologyViews(props.topology),
        props.nodeMask,
        props.edgeMask,
        props.order,
        props.parameters
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, resolution} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    validateNetworkViewBindings(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.nodeMask,
      props.edgeMask,
      props.order,
      props.parameters,
      output.counts,
      output.weightSums,
      output.maximumCount,
      output.maximumWeightSum
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
    validateNetworkViewBindings(id, graph, [], {
      textureConversion: output.texture ? resolution * resolution * 4 : 0,
      weightedBins: needsWeightBins(props) && !output.weightSums ? resolution * resolution * 4 : 0
    });
    const weightSums =
      output.weightSums ??
      (needsWeightBins(props)
        ? createTransientView(graph, `${id}-weight-sums-scratch`, 'uint32', resolution * resolution)
        : undefined);
    const operation = 'GPUAdjacencyMatrix';
    const directed = Boolean(props.directed);
    const mirror = !directed && Boolean(props.mirrorSlots);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const fill = (
      step: string,
      view: GraphDataView<'uint32'> | undefined,
      componentCount?: number
    ) => {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${step}`,
            operation,
            view,
            type: 'u32',
            value: '0u',
            componentCount
          })
        );
      }
    };
    fill('zero-counts', output.counts);
    fill('zero-weight-sums', weightSums);
    fill('zero-max-count', output.maximumCount, 1);
    fill('zero-max-weight-sum', output.maximumWeightSum, 1);

    const windowBinding: WGSLKernelBinding[] = props.parameters
      ? [
          {
            name: 'windowWords',
            view: props.parameters,
            type: 'u32',
            access: 'read'
          }
        ]
      : [];
    const windowSource = props.parameters
      ? `let rowStart = windowWords[windowWordsOffset];
  let rowEnd = windowWords[windowWordsOffset + 1u];
  let colStart = windowWords[windowWordsOffset + 2u];
  let colEnd = windowWords[windowWordsOffset + 3u];`
      : `let rowStart = 0u;
  let rowEnd = NODE_COUNT;
  let colStart = 0u;
  let colEnd = NODE_COUNT;`;
    // One kernel bins every requested statistic, so the CSR is walked once and the bin math is
    // shared. It needs one storage binding per optional input and output, so when that exceeds the
    // device limit each statistic gets its own kernel, as before.
    const addBinKernel = (
      step: string,
      targets: readonly {kind: 'count' | 'weight'; view: GraphDataView<'uint32'>}[]
    ) => {
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: neighbors,
          type: 'u32',
          access: 'read'
        },
        ...windowBinding
      ];
      if (props.order) {
        bindings.push({
          name: 'order',
          view: props.order,
          type: 'u32',
          access: 'read'
        });
      }
      if (props.nodeMask) {
        bindings.push({
          name: 'vertexMask',
          view: props.nodeMask,
          type: 'u32',
          access: 'read'
        });
      }
      if (props.edgeMask) {
        bindings.push({
          name: 'edgeMask',
          view: props.edgeMask,
          type: 'u32',
          access: 'read'
        });
      }
      if (targets.some(target => target.kind === 'weight')) {
        bindings.push({
          name: 'weights',
          view: weights!,
          type: 'f32',
          access: 'read'
        });
      }
      for (const {kind, view} of targets) {
        bindings.push({
          name: `${kind}Cells`,
          view,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      const amountSource = (kind: 'count' | 'weight') =>
        kind === 'count' ? '1u' : `fixedWeight(weights[weightsOffset + slot])`;
      const addSource = (cell: string) =>
        targets
          .map(
            ({kind}) =>
              `atomicAdd(&${kind}Cells[${kind}CellsOffset + ${cell}], ${amountSource(kind)});`
          )
          .join('\n      ');
      const mirrorSource = (cell: string) =>
        targets
          .map(
            ({kind}) =>
              `atomicAdd(&${kind}Cells[${kind}CellsOffset + ${cell}], ${amountSource(kind)});`
          )
          .join('\n      ');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation,
          variant: step,
          bindings,
          invocationCount: nodeCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const RESOLUTION: u32 = ${resolution}u;
const WEIGHT_SCALE: f32 = ${Math.fround(this.weightScale).toExponential(9)}f;
// Rounds the float32 product half up. The product feeds only floor() and comparisons, so no
// compiler can fuse it into an addition, and above 2^23 (where every float32 is an integer and
// adding 0.5 would round to the even neighbour) the product is returned as it is.
fn fixedWeight(weight: f32) -> u32 {
  let scaled = weight * WEIGHT_SCALE;
  if (!(scaled > 0.0)) {
    return 0u;
  }
  if (scaled >= 4294967040.0) {
    return 0xffffffffu;
  }
  let whole = floor(scaled);
  let roundsUp = whole < 8388608.0 && scaled >= whole + 0.5;
  return u32(whole) + select(0u, 1u, roundsUp);
}
fn getBin(position: u32, start: u32, end: u32) -> u32 {
  return (position - start) * RESOLUTION / (end - start);
}`,
          body: `${windowSource}
  if (${props.nodeMask ? 'vertexMask[vertexMaskOffset + index] == 0u' : 'false'}) {
    return;
  }
  let source = ${props.order ? 'order[orderOffset + index]' : 'index'};
  // A row contributes only from its own row position, or, when mirrored, from its column
  // position too. Skipping every other row keeps a zoomed window O(window rows), not O(slots).
  if (!((source >= rowStart && source < rowEnd)${mirror ? ' || (source >= colStart && source < colEnd)' : ''})) {
    return;
  }
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd2 = offsets[offsetsOffset + index + 1u];
  for (var slot = rowBegin; slot < rowEnd2; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= NODE_COUNT) {
      continue;
    }
    ${props.nodeMask ? 'if (vertexMask[vertexMaskOffset + neighbor] == 0u) { continue; }' : ''}
    ${props.edgeMask ? 'if (edgeMask[edgeMaskOffset + slot] == 0u) { continue; }' : ''}
    let targetPosition = ${props.order ? 'order[orderOffset + neighbor]' : 'neighbor'};
    if (source >= rowStart && source < rowEnd && targetPosition >= colStart && targetPosition < colEnd) {
      let cell = getBin(source, rowStart, rowEnd) * RESOLUTION + getBin(targetPosition, colStart, colEnd);
      ${addSource('cell')}
    }
    ${
      mirror
        ? `if (targetPosition != source && targetPosition >= rowStart && targetPosition < rowEnd && source >= colStart && source < colEnd) {
      let mirrorCell = getBin(targetPosition, rowStart, rowEnd) * RESOLUTION + getBin(source, colStart, colEnd);
      ${mirrorSource('mirrorCell')}
    }`
        : ''
    }
  }`
        })
      );
    };
    const binTargets: {kind: 'count' | 'weight'; view: GraphDataView<'uint32'>}[] = [
      {kind: 'count', view: output.counts}
    ];
    if (weightSums) {
      binTargets.push({kind: 'weight', view: weightSums});
    }
    const fusedBindingCount = getFusedBinBindingCount(props);
    if (
      binTargets.length === 1 ||
      fusedBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage
    ) {
      addBinKernel(binTargets.length === 1 ? 'bin-counts' : 'bin', binTargets);
    } else {
      addBinKernel('bin-counts', [binTargets[0]]);
      addBinKernel('bin-weights', [binTargets[1]]);
    }

    if (output.maximumCount || output.maximumWeightSum) {
      // One workgroup-local maximum per workgroup, then one global atomicMax per workgroup instead
      // of one per cell on a single address.
      const bindings: WGSLKernelBinding[] = [];
      let load = '';
      let reduce = '';
      let flush = '';
      let shared = '';
      if (output.maximumCount) {
        bindings.push(
          {name: 'counts', view: output.counts, type: 'u32', access: 'read'},
          {
            name: 'maximumCountOut',
            view: output.maximumCount,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        shared += 'var<workgroup> sharedCounts: array<u32, 256>;\n';
        load +=
          'sharedCounts[localInvocationIndex] = select(0u, counts[countsOffset + index], inRange);\n  ';
        reduce +=
          'sharedCounts[localInvocationIndex] = max(sharedCounts[localInvocationIndex], sharedCounts[localInvocationIndex + stride]);\n      ';
        flush += 'atomicMax(&maximumCountOut[maximumCountOutOffset], sharedCounts[0]);\n    ';
      }
      if (output.maximumWeightSum) {
        bindings.push(
          {
            name: 'sums',
            view: weightSums!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'maxSumOut',
            view: output.maximumWeightSum,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        shared += 'var<workgroup> sharedSums: array<u32, 256>;\n';
        load +=
          'sharedSums[localInvocationIndex] = select(0u, sums[sumsOffset + index], inRange);\n  ';
        reduce +=
          'sharedSums[localInvocationIndex] = max(sharedSums[localInvocationIndex], sharedSums[localInvocationIndex + stride]);\n      ';
        flush += 'atomicMax(&maxSumOut[maxSumOutOffset], sharedSums[0]);\n    ';
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-maxima`,
          operation,
          variant: 'maxima',
          bindings,
          invocationCount: resolution * resolution,
          guardIndex: false,
          declarations: shared,
          // No early return: every invocation of a workgroup must reach the barriers.
          body: `let inRange = index < INVOCATION_COUNT;
  ${load}workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      ${reduce}}
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    ${flush}}`
        })
      );
    }

    if (output.texture) {
      const source = output.textureStatistic === 'weightSum' ? weightSums! : output.counts;
      const asFloat = createTransientView(
        graph,
        `${id}-texture-values`,
        'float32',
        resolution * resolution
      );
      const scale = output.textureStatistic === 'weightSum' ? Math.fround(this.weightScale) : 1;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-texture-values`,
          operation,
          variant: 'texture-values',
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {
              name: 'valuesOut',
              view: asFloat,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: resolution * resolution,
          declarations: SCALED_QUOTIENT_WGSL,
          body: `valuesOut[valuesOutOffset + index] = getScaledQuotient(f32(source[sourceOffset + index]), ${scale.toExponential(9)}f, 1.0, 0);`
        })
      );
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: asFloat}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

/** Must match the optional read bindings and count/weight targets of the fused bin dispatch. */
function getFusedBinBindingCount(props: GPUAdjacencyMatrixProps): number {
  return (
    3 +
    Number(Boolean(props.parameters)) +
    Number(Boolean(props.order)) +
    Number(Boolean(props.nodeMask)) +
    Number(Boolean(props.edgeMask)) +
    2 * Number(needsWeightBins(props))
  );
}

function needsWeightBins(props: GPUAdjacencyMatrixProps): boolean {
  return Boolean(
    props.output.weightSums ||
      props.output.maximumWeightSum ||
      (props.output.texture && props.output.textureStatistic === 'weightSum')
  );
}
