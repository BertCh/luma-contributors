// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../../raster-analysis/raster-grid/raster-construction-snapshot';
import {
  validateTerrainBindings,
  getTerrainGateScratchByteLength
} from '../terrain-processing/terrain-binding-preflight';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createContributorTransientView} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {validateRasterIterations} from '../../raster-analysis/cost-distance/raster-relaxation';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
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

const OPERATION = 'GPUTerrainWatersheds';

/** Label of invalid cells, and of cells that drain to no pour point. */
export const GPU_TERRAIN_WATERSHED_NONE = 0xffffffff;

/**
 * Label of valid cells whose watershed is unknown: the D8 path meets a
 * `GPU_TERRAIN_FLOW_UNRESOLVED` direction before a stop cell, or pointer jumping stopped at
 * `maximumIterations` before the path reached its stop cell. A converged run on resolved directions
 * never writes it. Never a legitimate label: pour point indices and cell indices are smaller.
 */
export const GPU_TERRAIN_WATERSHED_UNRESOLVED = 0xfffffffe;

/** One-row status words of {@link GPUTerrainWatersheds}. */
export type GPUTerrainWatershedsStatus = {
  /** One row set to 1 when pointer jumping is stable and every valid path reaches a genuine stop. Presence is compile-time. */
  converged?: GraphDataView<'uint32'>;
  /** One row set to 1 when `maximumIterations` ran out before pointer jumping converged. Presence is compile-time. */
  iterationLimitReached?: GraphDataView<'uint32'>;
  /** One row: gated pointer-jumping rounds executed. Presence is compile-time. */
  iterationCount?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTerrainWatersheds}. */
export type GPUTerrainWatershedsOutput = {
  /**
   * With pour points: the index (into `pourPoints`) of the nearest pour point downstream of each
   * cell, inclusive, or `GPU_TERRAIN_WATERSHED_NONE` when the path reaches none. Without pour points:
   * the cell index of the terminal cell (outlet or pit) the cell drains to. Invalid cells are always
   * `GPU_TERRAIN_WATERSHED_NONE`. `GPU_TERRAIN_WATERSHED_UNRESOLVED` on valid cells whose path
   * meets an unresolved direction or a receiver cycle before a stop cell, or has not reached its
   * stop within the pointer-jumping budget. Other completed paths retain their labels.
   */
  labels: GraphDataView<'uint32'>;
  /** One-row status words. */
  status?: GPUTerrainWatershedsStatus;
};

/**
 * Properties for {@link GPUTerrainWatersheds}.
 *
 * Compile-time: `grid` size, `maximumIterations`, and the presence of `pourPoints` and `output.status`.
 * Per-frame: the contents of `flowDirections` and `pourPoints`. The contributor works on D8 receiver
 * indices only, so it has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing
 * give the same labels).
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainWatershedsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-watersheds'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid;
   * `GPU_TERRAIN_FLOW_UNRESOLVED` (0xfffffffe) unknown, which makes the labels that depend on it
   * `GPU_TERRAIN_WATERSHED_UNRESOLVED`. Any other code, and any code pointing outside the grid, is
   * treated as terminal. Typically `GPUTerrainFlow.flowDirections`. A grid in another numbering
   * needs `directionEncoding`. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default
   * pointer) or `'taudem'`, see {@link GPUTerrainFlowDirectionEncoding}. Zero and the two
   * sentinels mean the same in each. Compile-time.
   */
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  /**
   * Optional pour points as cell indices (`row * width + column`). Indices outside the grid and
   * indices of invalid cells are ignored. When several points share a cell the lowest point index
   * wins. Length is compile-time, contents per-frame. Without pour points the contributor labels
   * drainage basins.
   */
  pourPoints?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainWatershedsOutput;
  /**
   * Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells
   * need at most 32 doubling rounds. Compile-time.
   */
  maximumIterations?: number;
};

/**
 * Watershed delineation on one raster tile from D8 flow directions.
 *
 * Every cell follows its D8 receivers to a stop cell: a pour point when `pourPoints` is given, the
 * terminal cell otherwise. The path is resolved with GPU pointer jumping
 * (`pointer[c] = pointer[pointer[c]]` rounds), the same technique as parallel list ranking
 * (Wyllie 1979). Explicit budgets read a separate snapshot per round, making partial labels
 * deterministic. The default route updates in place; its completed acyclic fixpoint is unique.
 * Pour points
 * are marked with `atomicMin` so that duplicates resolve to the lowest point index. Nested pour
 * points give nested watersheds: each cell takes its nearest downstream pour point, and cells that
 * drain past all pour points are `GPU_TERRAIN_WATERSHED_NONE`.
 *
 * Without pour points the labels are drainage basins (the outlet or pit cell index), as in the
 * basin step of standard hydrologic toolboxes. Receiver cycles, possible only in caller-supplied
 * directions, leave `output.status.converged` at 0 and the labels of the affected cells at
 * `GPU_TERRAIN_WATERSHED_UNRESOLVED`.
 *
 * Compose it with {@link GPUTerrainFlow} through `flowDirections`. The default numbering is ESRI
 * (also WhiteboxTools with `--esri_pntr`); a default WhiteboxTools pointer grid needs
 * `directionEncoding: 'whitebox'` and a TauDEM D8 grid `'taudem'`. Without the right encoding a
 * Whitebox grid is read 45 degrees off with no error, because its codes are all valid ESRI codes.
 * In every numbering the codes are raster offsets ("south" is `row + 1`), not compass bearings:
 * the row sign of the grid affine is not read, so a tool's grid held with its rows flipped to
 * advance north must have its rows flipped back first, see {@link GPUTerrainFlowDirectionEncoding}.
 * Results are tile-local: watersheds do not continue across tile edges.
 */
export class GPUTerrainWatersheds implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainWatershedsProps;

  constructor(props: GPUTerrainWatershedsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'directionEncoding',
      'maximumIterations',
      'flowDirections',
      'pourPoints',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(props.output, ['labels', 'status']);
    props = {
      ...props,
      grid: snapshotRasterGrid(props.grid),
      output: Object.freeze({
        ...capturedOutput,
        status:
          capturedOutput.status &&
          snapshotConstructionFields(capturedOutput.status, [
            'converged',
            'iterationLimitReached',
            'iterationCount'
          ])
      })
    };
    this.id = props.id ?? 'terrain-watersheds';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterIterations(id, 'maximumIterations', props.maximumIterations ?? 32);
    validateTerrainFlowDirectionEncoding(id, props.directionEncoding);
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'output.labels', output.labels, 'uint32', cellCount);
    if (props.pourPoints) {
      validateSinglePackedView(props.pourPoints, ['uint32'], `${id} pourPoints`);
    }
    validateTerrainTracingStatus(id, output.status);
    const outputs = [
      output.labels,
      output.status?.converged,
      output.status?.iterationLimitReached,
      output.status?.iterationCount
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.flowDirections, props.pourPoints]);
    validateRasterBuffersDistinct(id, outputs, []);
  }

  /** Returns marker, pointer-jumping and labeling nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'pointer scratch': 4 * this.props.grid.width * this.props.grid.height,
      'gate table': getTerrainGateScratchByteLength(this.props.maximumIterations ?? 32)
    });
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.pourPoints,
      output.labels,
      output.status?.converged,
      output.status?.iterationLimitReached,
      output.status?.iterationCount
    ]);
    const cellCount = width * height;
    const pointer = createContributorTransientView(
      graph,
      'GPUTerrainWatersheds',
      id,
      `${id}-pointer`,
      'uint32',
      cellCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    let markers: GraphDataView<'uint32'> | undefined;
    if (props.pourPoints) {
      markers = createContributorTransientView(
        graph,
        'GPUTerrainWatersheds',
        id,
        `${id}-markers`,
        'uint32',
        cellCount
      );
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-markers-init`,
          operation: OPERATION,
          view: markers,
          type: 'u32',
          value: '0xffffffffu'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-markers-mark`,
          operation: OPERATION,
          variant: 'mark-pour-points',
          bindings: [
            {name: 'pourPoints', view: props.pourPoints, type: 'u32', access: 'read'},
            {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
            {name: 'markers', view: markers, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: props.pourPoints.length,
          declarations: `const CELL_COUNT: u32 = ${cellCount}u;`,
          body: `let cell = pourPoints[pourPointsOffset + index];
  if (cell < CELL_COUNT && flowDirections[flowDirectionsOffset + cell] != 0xffffffffu) {
    atomicMin(&markers[markersOffset + cell], index);
  }`
        })
      );
    }
    nodes.push(
      ...createTerrainFlowPointerNodes(graph, {
        id,
        contributorId: id,
        operation: OPERATION,
        width,
        height,
        maxIterations: props.maximumIterations ?? 32,
        stableRounds: props.maximumIterations !== undefined,
        flowDirections: props.flowDirections,
        directionEncoding: props.directionEncoding,
        pointer,
        stops: markers ? {kind: 'markers', markers} : {kind: 'none'},
        converged: output.status?.converged,
        iterationLimitReached: output.status?.iterationLimitReached,
        iterationCount: output.status?.iterationCount
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'pointer', view: pointer, type: 'u32', access: 'read'},
      ...(markers
        ? [{name: 'markers', view: markers, type: 'u32' as const, access: 'read' as const}]
        : []),
      ...(output.status?.converged
        ? [
            {
              name: 'converged',
              view: output.status.converged,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-labels`,
        operation: OPERATION,
        variant: markers ? 'label-pour-points' : 'label-basins',
        bindings,
        invocationCount: cellCount,
        declarations: getTerrainFlowTracingWGSL(width, height, props.directionEncoding),
        body: `let reached = pointer[pointerOffset + index];
  var label = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL) {
    ${
      markers
        ? `label = markers[markersOffset + reached];
    let isStop = label != NO_CELL;`
        : `if (flowDirections[flowDirectionsOffset + reached] != NO_CELL) { label = reached; }
    let isStop = false;`
    }
    // Doubling can collapse a receiver cycle to self-pointers. A genuine terminal must have
    // no original receiver, unless an explicit pour point stops the path there.
    let reachedCode = getFlowDirectionCode(flowDirections[flowDirectionsOffset + reached]);
    if (pointer[pointerOffset + reached] != reached ||
        (!isStop && (reachedCode == UNRESOLVED_DIRECTION ||
                     getFlowReceiver(reached, reachedCode) != NO_CELL))) {
      label = ${GPU_TERRAIN_WATERSHED_UNRESOLVED}u;
      ${output.status?.converged ? 'atomicStore(&converged[convergedOffset], 0u);' : ''}
    }
  }
  labels[labelsOffset + index] = label;`
      })
    );
    return nodes;
  }
}
