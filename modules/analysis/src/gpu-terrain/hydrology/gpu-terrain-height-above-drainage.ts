// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPU_RASTER_MAXIMUM_ITERATIONS} from '../../raster-analysis/cost-distance/raster-relaxation';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  createTerrainFractionalHANDNodes,
  TERRAIN_HAND_STATE_WORDS
} from './terrain-fractional-hand';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../../raster-analysis/raster-band-utils';
import {
  createTerrainFlowPointerNodes,
  getTerrainFlowTracingWGSL,
  validateTerrainFlowDirectionEncoding,
  validateTerrainTracingCellView,
  validateTerrainTracingStatus,
  type GPUTerrainFlowDirectionEncoding
} from './terrain-flow-tracing';
import {GET_QUIET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_TERRAIN_FLOW_STREAM_UNRESOLVED} from './gpu-terrain-flow';

const OPERATION = 'GPUTerrainHeightAboveDrainage';
// getRasterBufferBandNodes uses one published neighborhood or texture conversion node.
const RASTER_CANONICALIZATION_NODE_ALLOWANCE = 1;

/** Value of `drainageCells` for cells with no stream downstream, and for invalid cells. */
export const GPU_TERRAIN_DRAINAGE_NONE = 0xffffffff;

/**
 * Value of `drainageCells` for cells whose path meets a `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED`
 * stream word or a `GPU_TERRAIN_FLOW_UNRESOLVED` direction before a stream cell, or whose pointer
 * jumping or fractional dependency sweeps stopped at `maximumIterations` before finishing: the drainage cell is unknown, and
 * HAND is NaN. A converged run on resolved inputs never writes it.
 */
export const GPU_TERRAIN_DRAINAGE_UNRESOLVED = 0xfffffffe;

/** One-row iteration status; a round is a D8 doubling or a fractional dependency sweep. */
export type GPUTerrainHeightAboveDrainageStatus = GPUIterationStatusPort;

/** A resolved fractional path reaches more than one first stream cell. HAND is still valid. */
export const GPU_TERRAIN_DRAINAGE_MULTIPLE = 0xfffffffd;

/** Caller-owned outputs of {@link GPUTerrainHeightAboveDrainage}; at least one view is required. */
export type GPUTerrainHeightAboveDrainageOutput = {
  /**
   * D8 elevation difference, or fraction-weighted average vertical drop to first streams, in float32
   * elevation units. 0 on stream cells. NaN for invalid cells, cells whose path ends without
   * reaching a stream, cells whose path first meets an unresolved stream word, and cells whose
   * elevation or drainage elevation is not finite, and cells whose drainage cell is
   * `GPU_TERRAIN_DRAINAGE_UNRESOLVED`. Presence is compile-time.
   */
  heightAboveDrainage?: GraphDataView<'float32'>;
  /**
   * Cell index of the stream cell each cell drains to (the cell itself for stream cells),
   * `GPU_TERRAIN_DRAINAGE_UNRESOLVED` when the path first meets an unresolved stream word or
   * direction, or pointer jumping did not converge, or `GPU_TERRAIN_DRAINAGE_NONE`. Presence is
   * compile-time. In fractional mode, a unique first stream gives its cell index; multiple first
   * streams give `GPU_TERRAIN_DRAINAGE_MULTIPLE` (including upstream cells after a split).
   */
  drainageCells?: GraphDataView<'uint32'>;
  /** One-row status words. */
  status?: GPUTerrainHeightAboveDrainageStatus;
};

/**
 * Properties for {@link GPUTerrainHeightAboveDrainage}.
 *
 * Compile-time: `grid` size, `maximumIterations`, the elevation format and calibration, and which
 * optional outputs are provided. Per-frame: the contents of `elevation`, `flowDirections` and
 * `streams` and `flowFractions`. The routing choice is compile-time. It has no cell-size model: the result
 * is a vertical difference in elevation units, independent of `uniform`, `web-mercator` or
 * `geographic` cell spacing.
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainHeightAboveDrainageProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-height-above-drainage'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Routing surface whose heights are differenced: normally `GPUTerrainFlow.filledElevation`, or
   * the raw DEM. Validity, nodata, scale and offset are honored; cells without a finite calibrated
   * elevation get NaN. Format is compile-time, contents per-frame.
   */
  elevation: GPURasterBand;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid;
   * `GPU_TERRAIN_FLOW_UNRESOLVED` (0xfffffffe) unknown, which makes cells whose path meets it
   * before a stream cell unresolved. Any other code, and any code pointing outside the grid, is
   * treated as terminal. Typically `GPUTerrainFlow.flowDirections`. A grid in another numbering (a
   * default WhiteboxTools pointer, a TauDEM D8 grid) needs `directionEncoding`. Per-frame.
   */
  flowDirections?: GraphDataView<'uint32'>;
  /**
   * Alternative to `flowDirections`: eight flow fractions per cell, east first clockwise.
   * Use `GPUTerrainFlowFractions` with `routing: 'd-infinity'` and the same routing surface
   * and validity as `elevation`. Positive rows sum to one (tolerance 1e-5); zero rows are
   * outlets. NaN/negative/malformed rows or positive out-of-grid edges are unresolved.
   * Stream cells stop routing even when their fraction row is unresolved. Exactly one of
   * `flowDirections` and `flowFractions` is required. Contents per-frame, presence compile-time.
   *
   * A cell is NaN if any positive branch has no finite drop to a stream (TauDEM's default
   * edge-contamination check). Unknown branches propagate DRAINAGE_UNRESOLVED. Zero-weight
   * branches do not affect the result. MFD fractions also work with the same average algebra.
   */
  flowFractions?: GraphDataView<'float32'>;
  /**
   * Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default
   * pointer) or `'taudem'`, see {@link GPUTerrainFlowDirectionEncoding}. Zero and the two
   * sentinels mean the same in each. Compile-time.
   */
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  /**
   * Stream mask, one word per cell, non-zero for stream cells. Typically `GPUTerrainFlow.streams`,
   * whose `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` word (accumulation round limit hit) marks a cell of
   * unknown stream status. Per-frame.
   */
  streams: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainHeightAboveDrainageOutput;
  /**
   * Maximum gated rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells
   * need at most 32 doubling rounds. Fractional routing defaults to 128 dependency sweeps;
   * a path of depth D needs D sweeps plus one to certify convergence. Compile-time.
   */
  maximumIterations?: number;
};

/**
 * Height Above the Nearest Drainage (HAND) on one raster tile, following D8 or fractional flow.
 *
 * HAND is the elevation of a cell above the stream cell its flow path reaches (Renno et al. 2008;
 * Nobre et al. 2011). Here the drainage cell is the first stream cell on the cell's D8 path, found
 * by GPU pointer jumping: every cell points at its receiver (or itself when it is a stream cell or
 * has no receiver) and `pointer[c] = pointer[pointer[c]]` is repeated in place until nothing
 * changes. The fixpoint is the first stream cell downstream, or the terminal cell when there is
 * none; it is unique, so the result does not depend on thread scheduling. Heights are one float32
 * subtraction, so they are bit-exact against a sequential walk.
 *
 * With `flowFractions`, the TauDEM DinfDistDown vertical average recurrence is
 * `H[c] = sum(p[c,n] * (z[c] - z[n] + H[n])) / sum(p[c,n])`, stopping at streams (H = 0).
 * Synchronous GPU sweeps finalize cells once their downstream dependencies are known; each
 * invocation visits only eight neighbors. Separate read/write snapshots make values and budget
 * behavior deterministic. Unfinished cells publish NaN and DRAINAGE_UNRESOLVED; completed cells
 * remain available at the limit. Receiver cycles exhaust the budget and do not report convergence.
 * The result is a vertical difference in elevation units, not a path length.
 *
 * Compose it with {@link GPUTerrainFlow}: pass its `filledElevation` (or the raw DEM),
 * `flowDirections` and `streams`. It is a separate contributor so grids from other tools can be
 * used: name their numbering with `directionEncoding` (the default is ESRI). In every numbering
 * the codes are raster offsets ("south" is `row + 1`), not compass bearings: the row sign of the
 * grid affine is not read, see {@link GPUTerrainFlowDirectionEncoding}.
 */
export class GPUTerrainHeightAboveDrainage implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainHeightAboveDrainageProps;

  /** Upper bound including raster canonicalization and iteration status publication. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUTerrainHeightAboveDrainageProps) {
    this.id = props.id ?? 'terrain-height-above-drainage';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    const maximumIterations = props.maximumIterations ?? (props.flowFractions ? 128 : 32);
    validateCapacity(id, 'maximumIterations', maximumIterations, 1, GPU_RASTER_MAXIMUM_ITERATIONS);
    // Canonicalization uses at most one kernel. Fractional rounds use solve, copy and gate;
    // D8 uses one round and one gate per four rounds, plus reset/slots/status and final output.
    this.estimatedCommandNodeCount = props.flowFractions
      ? RASTER_CANONICALIZATION_NODE_ALLOWANCE + 4 + 3 * maximumIterations
      : RASTER_CANONICALIZATION_NODE_ALLOWANCE +
        6 +
        maximumIterations +
        Math.ceil(maximumIterations / 4);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRequirement(
      id,
      Boolean(props.flowDirections) !== Boolean(props.flowFractions),
      'flowDirections and flowFractions require exactly one routing input'
    );
    validateRequirement(
      id,
      !props.flowFractions || props.directionEncoding === undefined,
      'directionEncoding requires flowDirections'
    );
    validateTerrainFlowDirectionEncoding(id, props.directionEncoding);
    validateRequirement(
      id,
      Boolean(output.heightAboveDrainage || output.drainageCells),
      'output requires at least one output view'
    );
    if (props.flowDirections) {
      validateTerrainTracingCellView(
        id,
        'flowDirections',
        props.flowDirections,
        'uint32',
        cellCount
      );
    }
    if (props.flowFractions) {
      validateRequirement(
        id,
        props.flowFractions.format === 'float32',
        'flowFractions must use float32'
      );
      validateSinglePackedView(props.flowFractions, ['float32'], `${id} flowFractions`);
      validateViewLength(id, 'flowFractions', props.flowFractions, cellCount * 8);
    }
    validateTerrainTracingCellView(id, 'streams', props.streams, 'uint32', cellCount);
    if (output.heightAboveDrainage) {
      validateTerrainTracingCellView(
        id,
        'output.heightAboveDrainage',
        output.heightAboveDrainage,
        'float32',
        cellCount
      );
    }
    if (output.drainageCells) {
      validateTerrainTracingCellView(
        id,
        'output.drainageCells',
        output.drainageCells,
        'uint32',
        cellCount
      );
    }
    validateRequirement(
      id,
      props.elevation.storage.kind !== 'buffer' ||
        props.elevation.storage.values.length === cellCount,
      'elevation must contain one value per cell'
    );
    validateTerrainTracingStatus(id, output.status);
    const outputs = [
      output.heightAboveDrainage,
      output.drainageCells,
      output.status?.converged,
      output.status?.iterationLimitReached,
      output.status?.iterationCount
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      ...getRasterBandViews(props.elevation),
      props.flowDirections,
      props.flowFractions,
      props.streams
    ]);
    validateRasterBuffersDistinct(id, outputs, []);
  }

  /** Returns canonicalization, pointer-jumping and HAND nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const cellCount = width * height;
    // The fractional snapshot packs [state, height bits, drainage cell] for each cell.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'HAND scratch',
      cellCount * (props.flowFractions ? TERRAIN_HAND_STATE_WORDS * 4 : 4),
      'reduce the grid dimensions'
    );
    if (props.flowFractions) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'flowFractions',
        cellCount * 32,
        'reduce the grid dimensions'
      );
    }
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.flowFractions,
      props.streams,
      output.heightAboveDrainage,
      output.drainageCells,
      output.status?.converged,
      output.status?.iterationLimitReached,
      output.status?.iterationCount
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    let elevation: GraphDataView<'float32'> | undefined;
    if (output.heightAboveDrainage || props.flowFractions) {
      const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
      nodes.push(...source.nodes);
      elevation = source.band.storage.values as GraphDataView<'float32'>;
    }
    if (props.flowFractions && elevation) {
      return [
        ...nodes,
        ...createTerrainFractionalHANDNodes(graph, {
          id,
          grid: props.grid,
          elevation,
          fractions: props.flowFractions,
          streams: props.streams,
          output,
          maximumIterations: props.maximumIterations ?? 128
        })
      ];
    }
    // Construction requires exactly one routing input; fractional mode returned above.
    const flowDirections = props.flowDirections!;
    // The pointer stays untouched by the final kernel, which reads other cells' pointers to check
    // that each path reached its fixpoint, so drainage cells are written to their own view.
    const pointer = createTransientView(graph, `${id}-pointer`, 'uint32', cellCount);
    nodes.push(
      ...createTerrainFlowPointerNodes(graph, {
        id,
        contributorId: id,
        operation: OPERATION,
        width,
        height,
        maxIterations: props.maximumIterations ?? 32,
        flowDirections,
        directionEncoding: props.directionEncoding,
        pointer,
        stops: {kind: 'streams', streams: props.streams},
        converged: output.status?.converged,
        iterationLimitReached: output.status?.iterationLimitReached,
        iterationCount: output.status?.iterationCount
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {name: 'flowDirections', view: flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'pointer', view: pointer, type: 'u32', access: 'read'}
    ];
    if (output.drainageCells) {
      bindings.push({
        name: 'drainageOut',
        view: output.drainageCells,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (elevation && output.heightAboveDrainage) {
      bindings.push(
        {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
        {name: 'heightOut', view: output.heightAboveDrainage, type: 'f32', access: 'read_write'}
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-height`,
        operation: OPERATION,
        variant: 'height-above-drainage',
        bindings,
        invocationCount: cellCount,
        declarations: `${getTerrainFlowTracingWGSL(width, height)}
${GET_QUIET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`,
        body: `let reached = pointer[pointerOffset + index];
  var drainage = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL &&
      flowDirections[flowDirectionsOffset + reached] != NO_CELL) {
    if (pointer[pointerOffset + reached] != reached) {
      // Pointer jumping stopped early: 'reached' is an intermediate cell, not the first stream.
      drainage = ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u;
    } else if (streams[streamsOffset + reached] != 0u) {
      // An unresolved stream word stops the path too: the first stream may be that cell or lie
      // further down, so the drainage cell is unknown rather than wrong.
      drainage = select(reached, ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u,
        streams[streamsOffset + reached] == ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u);
    } else if (flowDirections[flowDirectionsOffset + reached] == UNRESOLVED_DIRECTION ||
        flowDirections[flowDirectionsOffset + index] == UNRESOLVED_DIRECTION) {
      // The path ends on a cell whose direction is unknown, not on an outlet or a sink.
      drainage = ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u;
    }
  }
  ${
    elevation
      ? `var result = getQuietNaN();
  if (drainage < ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u) {
    let cellElevation = elevation[elevationOffset + index];
    let drainageElevation = elevation[elevationOffset + drainage];
    if (isFiniteValue(cellElevation) && isFiniteValue(drainageElevation)) {
      result = cellElevation - drainageElevation;
    }
  }
  heightOut[heightOutOffset + index] = result;`
      : ''
  }
  ${output.drainageCells ? 'drainageOut[drainageOutOffset + index] = drainage;' : ''}`
      })
    );
    return nodes;
  }
}
