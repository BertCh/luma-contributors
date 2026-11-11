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
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../raster-analysis/raster-band-utils';
import {
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  FIXED_POINT_WGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
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
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

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
  /** Dense zero-based basin IDs; 0xffffffff excludes a cell. */
  basinIds: GraphDataView<'uint32'>;
  basinCount: number;
  /**
   * Stream mask: a nonzero word is a stream cell (so a mask of Strahler orders works), 0 is not.
   * The unresolved sentinels `0xffffffff` and `0xfffffffe` are not stream cells: they add no
   * stream length and their profile status is 2.
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
  };
};

/**
 * Composes sorted basin reductions and pointer-jumped stream profiles. Each edge belongs to its
 * source cell's basin.
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
    this.id = props.id ?? 'terrain-catchment-summary';
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateTerrainFlowDirectionEncoding(this.id, props.directionEncoding);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateCapacity(this.id, 'basinCount', props.basinCount);
    validateCapacity(this.id, 'histogramBins', props.histogramBins);
    validateCapacity(this.id, 'maximumSteps', props.maximumSteps ?? count);
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
    checkTerrainViews(
      this.id,
      [
        props.elevation,
        props.basinIds,
        props.streams,
        props.flowDirections,
        ...(props.validity ? [props.validity] : [])
      ],
      Object.values(props.output)
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
      ...Object.values(props.output)
    ]);
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
    // A stream cell is any nonzero word except the two unresolved sentinels, the test of stream
    // order, stream network, flow distance and height above drainage.
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
    // At most one term per cell enters any sum of a basin.
    const headroomBits = getFixedPointHeadroomBits(count);
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
    const ranges = createTransientView(graph, `${id}-ranges`, 'uint32', props.basinCount * 2);
    const states = [0, 1].map(side =>
      createTransientView(graph, `${id}-profiles-${side}`, 'uint32', count * 4)
    );
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
        body: `let basin = basins[basinsOffset + index]; keys[keysOffset + index] = select(0xffffffffu, basin, basin < ${props.basinCount}u && isFiniteValue(elevation[elevationOffset + index])); cells[cellsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-basin-sort`,
        keys,
        values: cells,
        outputKeys: sortedKeys,
        outputValues: order
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-ranges-initialize`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: props.basinCount,
        bindings: [terrainBinding('ranges', ranges, 'read_write')],
        body: 'ranges[rangesOffset + index * 2u] = 0u; ranges[rangesOffset + index * 2u + 1u] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-ranges`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        bindings: [
          terrainBinding('keys', sortedKeys),
          terrainBinding('ranges', ranges, 'read_write')
        ],
        body: `let basin = keys[keysOffset + index]; if (basin >= ${props.basinCount}u) { return; }
if (index == 0u || keys[keysOffset + index - 1u] != basin) { ranges[rangesOffset + basin * 2u] = index; }
if (index + 1u == ${count}u || keys[keysOffset + index + 1u] != basin) { ranges[rangesOffset + basin * 2u + 1u] = index + 1u; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-statistics`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: props.basinCount,
        declarations: `${declarations}\n${FIXED_POINT_WGSL}`,
        bindings: [
          ...inputs,
          terrainBinding('order', order),
          terrainBinding('ranges', ranges),
          terrainBinding('statistics', props.output.statistics, 'read_write'),
          terrainBinding('hypsometry', props.output.hypsometry, 'read_write')
        ],
        // Two walks over the basin's cells: the first finds the count, the extrema and the largest
        // exponent of each kind of term, which sets the fixed-point unit; the second adds the terms
        // as integers in that unit. No float32 partial sum exists, so nothing rounds per cell.
        body: `var sampleCount = 0u; var lower = infinity(); var upper = -lower;
var elevationExponent = 0u; var areaExponent = 0u; var lengthExponent = 0u;
for (var entry = ranges[rangesOffset + index * 2u]; entry < ranges[rangesOffset + index * 2u + 1u]; entry++) {
 let cell = order[orderOffset + entry];
 let value = elevation[elevationOffset + cell];
 if (basins[basinsOffset + cell] != index || !isFiniteValue(value)) { continue; }
 sampleCount++; lower = min(lower, value); upper = max(upper, value);
 elevationExponent = max(elevationExponent, getFixedPointExponentBits(value));
 ${isUniform ? '' : 'areaExponent = max(areaExponent, getFixedPointExponentBits(getGroundCellArea(cell)));'}
 lengthExponent = max(lengthExponent, getFixedPointExponentBits(getStreamEdgeLength(cell)));
}
let elevationUnit = getFixedPointUnitExponent(elevationExponent, ${headroomBits}u);
let areaUnit = getFixedPointUnitExponent(areaExponent, ${headroomBits}u);
let lengthUnit = getFixedPointUnitExponent(lengthExponent, ${headroomBits}u);
var elevationSum = vec3<u32>(0u); var areaSum = vec3<u32>(0u); var lengthSum = vec3<u32>(0u);
for (var entry = ranges[rangesOffset + index * 2u]; entry < ranges[rangesOffset + index * 2u + 1u]; entry++) {
 let cell = order[orderOffset + entry];
 let value = elevation[elevationOffset + cell];
 if (basins[basinsOffset + cell] != index || !isFiniteValue(value)) { continue; }
 elevationSum = addFixedPoint(elevationSum, quantizeFixedPoint(value, elevationUnit));
 ${isUniform ? '' : 'areaSum = addFixedPoint(areaSum, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));'}
 lengthSum = addFixedPoint(lengthSum, quantizeFixedPoint(getStreamEdgeLength(cell), lengthUnit));
}
let sum = getFixedPointFloat(elevationSum, elevationUnit);
let streamLength = getFixedPointFloat(lengthSum, lengthUnit);
let area = ${isUniform ? `f32(sampleCount) * ${area}` : 'getFixedPointFloat(areaSum, areaUnit)'};
let offset = statisticsOffset + index * 6u;
statistics[offset] = area; statistics[offset + 1u] = select(getNaN(), sum / f32(max(sampleCount, 1u)), sampleCount > 0u);
statistics[offset + 2u] = select(getNaN(), lower, sampleCount > 0u); statistics[offset + 3u] = select(getNaN(), upper, sampleCount > 0u);
statistics[offset + 4u] = streamLength; statistics[offset + 5u] = select(getNaN(), streamLength / max(area, 1e-30), area > 0.0);
for (var bin = 0u; bin < ${props.histogramBins}u; bin++) {
 let threshold = lower + (upper - lower) * f32(bin + 1u) / ${getWGSLFloatLiteral(props.histogramBins)};
 var belowCount = 0u; var belowArea = vec3<u32>(0u);
 for (var entry = ranges[rangesOffset + index * 2u]; entry < ranges[rangesOffset + index * 2u + 1u]; entry++) {
  let cell = order[orderOffset + entry];
  let value = elevation[elevationOffset + cell];
  if (basins[basinsOffset + cell] == index && isFiniteValue(value) && (value <= threshold || bin + 1u == ${props.histogramBins}u)) { belowCount++; ${isUniform ? '' : 'belowArea = addFixedPoint(belowArea, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));'} }
 }
 hypsometry[hypsometryOffset + index * ${props.histogramBins}u + bin] = ${isUniform ? `f32(belowCount) * ${area}` : 'getFixedPointFloat(belowArea, areaUnit)'};
}`
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
          terrainBinding('status', props.output.profileStatus, 'read_write')
        ],
        body: `let center = elevation[elevationOffset + index]; let offset = stateOffset + index * 4u;
let distance = bitcast<f32>(state[offset + 1u]); var gradient = 0.0;
var result = select(2u, 1u, state[offset] == NO_CELL && state[offset + 3u] == 1u && state[offset + 2u] < ${props.maximumSteps ?? count}u && isFiniteValue(distance));
if (!isFiniteValue(center) || basins[basinsOffset + index] >= ${props.basinCount}u || !isStreamWord(streams[streamsOffset + index])) { result = select(0u, 2u, streams[streamsOffset + index] >= UNRESOLVED_DIRECTION); }
let receiver = getCellReceiver(index);
if (receiver != NO_CELL && isStreamWord(streams[streamsOffset + receiver])) { gradient = (center - elevation[elevationOffset + receiver]) / edgeLength(index, receiver); }
status[statusOffset + index] = result;
profiles[profilesOffset + index * 3u] = select(getNaN(), distance, result == 1u);
profiles[profilesOffset + index * 3u + 1u] = center;
profiles[profilesOffset + index * 3u + 2u] = select(getNaN(), gradient, result == 1u);`
      })
    );
    return nodes;
  }
}
