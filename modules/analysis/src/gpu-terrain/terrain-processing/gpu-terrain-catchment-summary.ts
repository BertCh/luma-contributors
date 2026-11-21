// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRasterCountNodes} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../raster-analysis/raster-band-utils';
import {
  validateGraphPorts,
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSortedSegmentOffsetsNode} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  createTerrainCatchmentReductionNodes,
  validateTerrainCatchmentReductionSizes,
  validateTerrainCatchmentReductionBindings
} from './terrain-catchment-reductions';
import {
  getTerrainFlowTracingWGSL,
  validateTerrainFlowDirectionEncoding,
  type GPUTerrainFlowDirectionEncoding
} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  createTerrainValidityMask,
  getTerrainGroundMeasureWGSL,
  terrainBinding,
  validateTerrainValidityView,
  TERRAIN_FINITE_WGSL,
  validateTerrainStatusPorts
} from './terrain-processing-utils';

import {
  createTerrainResolutionStatusNodes,
  getTerrainResolutionStatusViews,
  type TerrainResolutionStatusPort
} from './terrain-resolution-status';

export type GPUTerrainCatchmentSummaryProps = {
  id?: string;
  /**
   * Grid. Areas and lengths are in squared and plain affine coordinate units by default
   * (geographic degrees are not ground meters), or in square and plain ground meters under
   * `cellSizeMode`. Elevation, gradient and hypsometry elevations keep the elevation's units;
   * the gradient is elevation units per length unit.
   */
  grid: GPURasterMetadata;
  /**
   * Ground model for areas, stream lengths, profile distances and gradients. Default
   * `'uniform'` (affine units). `'web-mercator'` and `'geographic'` give ground meters and need
   * an axis-aligned grid. Compile-time.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  elevation: GraphDataView<'float32'>;
  /** Optional per-cell validity of `elevation` (uint32, nonzero valid); a zero word counts as NaN, so the cell is excluded. */
  validity?: GraphDataView<'uint32'>;
  /** Dense zero-based basin IDs; a value of `basinCount` or more (such as 0xffffffff) excludes a cell. */
  basinIds: GraphDataView<'uint32'>;
  basinCount: number;
  /**
   * Stream mask: a nonzero word is a stream cell (so a mask of Strahler orders works), 0 is not.
   * The unresolved sentinels `0xffffffff` and `0xfffffffe` are not stream cells: they add no
   * stream length and their profile status is 2. Thus membership is 1..0xfffffffd, unlike
   * StreamOrder, where 0xfffffffe is also membership. Translate imported unresolved 0xfffffffe
   * to StreamOrder's unknown 0xffffffff when sharing an imported mask with that consumer.
   */
  streams: GraphDataView<'uint32'>;
  /**
   * D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal). A
   * code without a receiver ends the stream there.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default
   * pointer) or `'taudem'`, see `GPUTerrainFlowDirectionEncoding`. Compile-time.
   */
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  histogramBins: number;
  maximumSteps?: number;
  output: {
    /** Six values per basin: valid area, mean, minimum, maximum, stream length, drainage density. */
    statistics: GraphDataView<'float32'>;
    /** Cumulative area by normalized elevation bin (upper edges). Last bin is total valid area. */
    hypsometry: GraphDataView<'float32'>;
    /** Three values per cell: downstream stream distance, elevation, outgoing elevation gradient. */
    profiles: GraphDataView<'float32'>;
    /** Per cell: 0 excluded/invalid, 1 complete profile, 2 cycle or exhausted walk. */
    profileStatus: GraphDataView<'uint32'>;
    /**
     * Optional aggregate profile diagnostics. `invalidCount` counts state 0 (excluded/invalid);
     * `converged` is 1 if no profile remains in state 2. `iterationLimitReached` is 1 when a
     * profile exhausts maximumSteps or the pointer-jump round budget (including cycles), not
     * merely because a dependency is invalid. Empty/excluded domains converge.
     */
    status?: TerrainResolutionStatusPort;
  };
};

/** Flattens named scalar ports alongside the per-cell outputs. */
function getOutputViews(output: GPUTerrainCatchmentSummaryProps['output']): GraphDataView[] {
  return [
    output.statistics,
    output.hypsometry,
    output.profiles,
    output.profileStatus,
    ...getTerrainResolutionStatusViews(output.status)
  ];
}

/**
 * Composes sorted basin reductions and pointer-jumped stream profiles. Each edge belongs to its
 * source cell's basin. Statistics and hypsometry reduce sorted runs in tiles of at most 256
 * cells per invocation, then join partials with exact fixed-point atomic additions. No invocation
 * walks a whole basin; a dominant basin spreads over as many tiles as it needs.
 * Excluded rows (invalid elevation/validity or out-of-range basin) have profile status 0,
 * even when their stream membership is unknown; an entirely excluded domain converges.
 *
 * The per-basin sums (elevation for the mean, ground area and hypsometry areas under
 * `cellSizeMode`, stream length) are fixed-point sums (`fixed-point-accumulator.ts`): every term
 * is added as an integer in units set by the largest term of the basin, so the sum does not depend
 * on the number of cells and the result is the float32 nearest the exact sum of the float32 terms.
 * A serial float32 sum would be 0.8 % off for a million cells at a constant 2000.
 */
export class GPUTerrainCatchmentSummary implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainCatchmentSummaryProps) {
    validateGraphPorts(props.id ?? 'terrain-catchment-summary', props, {
      elevation: {required: true, viewType: 'data', formats: ['float32']},
      validity: {required: false, viewType: 'data', formats: ['uint32']},
      basinIds: {required: true, viewType: 'data', formats: ['uint32']},
      streams: {required: true, viewType: 'data', formats: ['uint32']},
      flowDirections: {required: true, viewType: 'data', formats: ['uint32']},
      'output.statistics': {required: true, viewType: 'data', formats: ['float32']},
      'output.hypsometry': {required: true, viewType: 'data', formats: ['float32']},
      'output.profiles': {required: true, viewType: 'data', formats: ['float32']},
      'output.profileStatus': {required: true, viewType: 'data', formats: ['uint32']},
      'output.status.converged': {required: false, viewType: 'data', formats: ['uint32']},
      'output.status.invalidCount': {required: false, viewType: 'data', formats: ['uint32']},
      'output.status.iterationLimitReached': {
        required: false,
        viewType: 'data',
        formats: ['uint32']
      }
    });
    this.id = props.id ?? 'terrain-catchment-summary';
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateTerrainFlowDirectionEncoding(this.id, props.directionEncoding);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateCapacity(this.id, 'basinCount', props.basinCount);
    validateCapacity(this.id, 'histogramBins', props.histogramBins);
    validateCapacity(this.id, 'maximumSteps', props.maximumSteps ?? count);
    validateTerrainCatchmentReductionSizes(this.id, count, props.basinCount, props.histogramBins);
    validateRequirement(
      this.id,
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      ) > 0,
      'grid affine must have a nonzero determinant'
    );
    validateViewLength(this.id, 'elevation', props.elevation, count);
    validateViewLength(this.id, 'basinIds', props.basinIds, count);
    validateViewLength(this.id, 'streams', props.streams, count);
    validateViewLength(this.id, 'flowDirections', props.flowDirections, count);
    validateTerrainValidityView(this.id, props.validity, count);
    validateViewLength(this.id, 'output.statistics', props.output.statistics, props.basinCount * 6);
    validateViewLength(
      this.id,
      'output.hypsometry',
      props.output.hypsometry,
      props.basinCount * props.histogramBins
    );
    validateViewLength(this.id, 'output.profiles', props.output.profiles, count * 3);
    validateViewLength(this.id, 'output.profileStatus', props.output.profileStatus, count);
    validateTerrainStatusPorts(this.id, {...props.output.status});
    checkTerrainViews(
      this.id,
      [
        props.elevation,
        props.basinIds,
        props.streams,
        props.flowDirections,
        ...(props.validity ? [props.validity] : [])
      ],
      getOutputViews(props.output),
      [
        {writes: [props.output.statistics]},
        {writes: [props.output.hypsometry]},
        {
          reads: [
            ...(!props.validity ? [props.elevation] : []),
            props.basinIds,
            props.streams,
            props.flowDirections
          ],
          writes: [props.output.profiles, props.output.profileStatus]
        },
        ...getTerrainResolutionStatusViews(props.output.status).map(view => ({
          reads: view === props.output.status?.invalidCount ? [props.output.profileStatus] : [],
          writes: [view]
        }))
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    checkTerrainGraph(id, graph, [
      props.elevation,
      ...(props.validity ? [props.validity] : []),
      props.basinIds,
      props.streams,
      props.flowDirections,
      ...getOutputViews(props.output)
    ]);
    validateTerrainCatchmentReductionBindings(
      graph,
      id,
      count,
      props.basinCount,
      props.histogramBins
    );
    const masked = createTerrainValidityMask(graph, {
      id,
      operation: 'GPUTerrainCatchmentSummary',
      values: props.elevation,
      validity: props.validity
    });
    const elevation = masked.values;
    const isUniform = (props.cellSizeMode ?? 'uniform') === 'uniform';
    const area = getWGSLFloatLiteral(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      )
    );
    // Catchment profiles reserve both high stream words. StreamOrder and D8 HAND instead
    // reserve only 0xffffffff at their stream ports; their direction-port codes are separate.
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height, props.directionEncoding)}\n${TERRAIN_FINITE_WGSL}
${getTerrainGroundMeasureWGSL(props.grid, props.cellSizeMode)}
fn edgeLength(cell: u32, receiver: u32) -> f32 { return getGroundEdgeLength(cell, receiver); }
fn isStreamWord(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }
fn getCellReceiver(cell: u32) -> u32 { return getFlowReceiver(cell, getFlowDirectionCode(directions[directionsOffset + cell])); }
// Length of the stream edge leaving 'cell', 0 when the cell or its receiver is not a stream cell.
fn getStreamEdgeLength(cell: u32) -> f32 {
 let receiver = getCellReceiver(cell);
 if (isStreamWord(streams[streamsOffset + cell]) && receiver != NO_CELL && isStreamWord(streams[streamsOffset + receiver]) && isFiniteValue(elevation[elevationOffset + receiver])) { return edgeLength(cell, receiver); }
 return 0.0;
}`;
    const inputs = [
      terrainBinding('elevation', elevation),
      terrainBinding('basins', props.basinIds),
      terrainBinding('streams', props.streams),
      terrainBinding('directions', props.flowDirections)
    ];
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', count);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', count);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count);
    const order = createTransientView(graph, `${id}-order`, 'uint32', count);
    const ranges = createTransientView(graph, `${id}-ranges`, 'uint32', props.basinCount + 1);
    const states = [0, 1].map(side =>
      createTransientView(graph, `${id}-profiles-${side}`, 'uint32', count * 4)
    );
    const profileLimitFlags = props.output.status?.iterationLimitReached
      ? createTransientView(graph, `${id}-profile-limit-flags`, 'uint32', count)
      : undefined;
    const profileLimitCount = createTransientView(graph, `${id}-profile-limit-count`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...masked.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-basin-keys`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('elevation', elevation),
          terrainBinding('basins', props.basinIds),
          terrainBinding('keys', keys, 'read_write'),
          terrainBinding('cells', cells, 'read_write')
        ],
        body: `let basin = basins[basinsOffset + index]; keys[keysOffset + index] = select(${props.basinCount}u, basin, basin < ${props.basinCount}u && isFiniteValue(elevation[elevationOffset + index])); cells[cellsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-basin-sort`,
        keys,
        values: cells,
        outputKeys: sortedKeys,
        outputValues: order,
        // Keys run from 0 through basinCount, the key of an excluded cell.
        keyBits: getSortKeyBits(props.basinCount)
      }).getCommandNodes(graph),
      createSortedSegmentOffsetsNode(graph, {
        id: `${id}-ranges`,
        operation: 'GPUTerrainCatchmentSummary',
        segmentCount: props.basinCount,
        sortedKeys,
        segmentOffsets: ranges
      }),
      ...createTerrainCatchmentReductionNodes(graph, {
        id,
        cellCount: count,
        basinCount: props.basinCount,
        histogramBins: props.histogramBins,
        isUniform,
        area,
        declarations,
        inputs,
        order,
        ranges,
        statistics: props.output.statistics,
        hypsometry: props.output.hypsometry
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-profile-seeds`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations,
        bindings: [...inputs, terrainBinding('state', states[0], 'read_write')],
        body: `var parent = NO_CELL; var distance = 0.0; var steps = 0u; var result = 2u;
let code = getFlowDirectionCode(directions[directionsOffset + index]);
if (isFiniteValue(elevation[elevationOffset + index]) && isStreamWord(streams[streamsOffset + index])) {
 if (code == 0u) { result = 1u; } else {
  let receiver = getFlowReceiver(index, code);
  if (receiver != NO_CELL && isFiniteValue(elevation[elevationOffset + receiver]) && streams[streamsOffset + receiver] < UNRESOLVED_DIRECTION) {
   if (streams[streamsOffset + receiver] == 0u) { result = 1u; }
   else { parent = receiver; distance = edgeLength(index, receiver); steps = 1u; }
  }
 }
}
state[stateOffset + index * 4u] = parent; state[stateOffset + index * 4u + 1u] = bitcast<u32>(distance);
state[stateOffset + index * 4u + 2u] = steps; state[stateOffset + index * 4u + 3u] = result;`
      })
    ];
    const rounds = Math.ceil(Math.log2(Math.max(count, 1)));
    for (let iteration = 0; iteration < rounds; iteration++)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-profile-jump-${iteration}`,
          operation: 'GPUTerrainCatchmentSummary',
          invocationCount: count,
          bindings: [
            terrainBinding('previous', states[iteration % 2]),
            terrainBinding('next', states[(iteration + 1) % 2], 'read_write')
          ],
          body: `let offset = previousOffset + index * 4u; let destination = nextOffset + index * 4u;
for (var component = 0u; component < 4u; component++) { next[destination + component] = previous[offset + component]; }
let parent = previous[offset];
if (parent != 0xffffffffu) {
 let ancestor = previousOffset + parent * 4u;
 next[destination] = previous[ancestor];
 next[destination + 1u] = bitcast<u32>(bitcast<f32>(previous[offset + 1u]) + bitcast<f32>(previous[ancestor + 1u]));
 next[destination + 2u] = min(${count}u, previous[offset + 2u] + previous[ancestor + 2u]);
 next[destination + 3u] = previous[ancestor + 3u];
}`
        })
      );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-profiles`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('state', states[rounds % 2]),
          terrainBinding('profiles', props.output.profiles, 'read_write'),
          terrainBinding('status', props.output.profileStatus, 'read_write'),
          ...(profileLimitFlags
            ? [terrainBinding('limitFlags', profileLimitFlags, 'read_write')]
            : [])
        ],
        body: `let center = elevation[elevationOffset + index]; let offset = stateOffset + index * 4u;
let distance = bitcast<f32>(state[offset + 1u]); var gradient = 0.0;
var result = select(2u, 1u, state[offset] == NO_CELL && state[offset + 3u] == 1u && state[offset + 2u] < ${props.maximumSteps ?? count}u && isFiniteValue(distance));
if (!isFiniteValue(center) || basins[basinsOffset + index] >= ${props.basinCount}u) { result = 0u; }
else if (!isStreamWord(streams[streamsOffset + index])) { result = select(0u, 2u, streams[streamsOffset + index] >= UNRESOLVED_DIRECTION); }
let receiver = getCellReceiver(index);
if (receiver != NO_CELL && isStreamWord(streams[streamsOffset + receiver])) { gradient = (center - elevation[elevationOffset + receiver]) / edgeLength(index, receiver); }
status[statusOffset + index] = result;
${
  profileLimitFlags
    ? `let eligible = isFiniteValue(center) && basins[basinsOffset + index] < ${props.basinCount}u && isStreamWord(streams[streamsOffset + index]);
let exhausted = state[offset] != NO_CELL || (state[offset + 3u] == 1u && state[offset + 2u] >= ${props.maximumSteps ?? count}u);
limitFlags[limitFlagsOffset + index] = u32(result == 2u && eligible && exhausted);`
    : ''
}
profiles[profilesOffset + index * 3u] = select(getNaN(), distance, result == 1u);
profiles[profilesOffset + index * 3u + 1u] = center;
profiles[profilesOffset + index * 3u + 2u] = select(getNaN(), gradient, result == 1u);`
      })
    );
    if (profileLimitFlags) {
      nodes.push(
        ...createRasterCountNodes(graph, {
          id: `${id}-profile-limit-count`,
          operation: 'GPUTerrainCatchmentSummary',
          values: profileLimitFlags,
          count,
          condition: 'value != 0u',
          output: profileLimitCount
        })
      );
    }
    nodes.push(
      ...createTerrainResolutionStatusNodes(graph, {
        id,
        operation: 'GPUTerrainCatchmentSummary',
        states: props.output.profileStatus,
        status: props.output.status,
        iterationLimit: {source: profileLimitCount, expression: 'u32(source[sourceOffset] != 0u)'}
      })
    );
    return nodes;
  }
}
