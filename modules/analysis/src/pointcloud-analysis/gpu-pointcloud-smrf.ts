// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GPUReduction} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS
} from './pointcloud-limits';
import {getPointCloudDiamondMorphologyNodes} from './pointcloud-diamond-morphology';
import {createPointCloudGridIndex, getPointCloudRowDirection} from './pointcloud-grid-index';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {getPointCloudCellSource, validatePointCloudGrid} from './gpu-pointcloud-surfaces';
import type {PointCloudRasterGrid} from './gpu-pointcloud-surfaces';

export type GPUPointCloudSMRFProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  classifications: GraphDataView<'uint32'>;
  /** Caller-selected returns/class-bit/ignore policy. Zero rows retain their original class. */
  selection?: GraphDataView<'uint32'>;
  /** Set origin to cloud XY minimum and dimensions to floor(extent / cellSize) + 1 for PDAL grid alignment. */
  grid: PointCloudRasterGrid;
  slope?: number;
  scalar?: number;
  threshold?: number;
  /** Maximum physical window radius; defaults to 18 * cellSize; maximum 64 cells. */
  window?: number;
  /** Physical net-cut spacing, zero disables; maximum 32 cells. */
  cut?: number;
  groundClass?: number;
  otherClass?: number;
  onlyGround?: boolean;
  /** Provisional inpainted terrain in row-major cell order; zero without support. */
  groundSurface: GraphDataView<'float32'>;
  groundMask: GraphDataView<'uint32'>;
  outputClassifications: GraphDataView<'uint32'>;
  /**
   * Per-point bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_SMRF_STATUS_INVALID`): the point is
   * outside the grid or not finite. Bit 2 (`..._NO_TERRAIN`): its cell has no terrain anchors, so
   * it is not classified. Bit 4 (`..._EXCLUDED`): `selection` is zero for the point, which keeps its
   * original class. Bit 8 (`..._INPAINTED`): the terrain under the point is inpainted, or the point
   * lies within the filter margin of the grid edge, so its classification is less certain. Bits 1,
   * 2 and 4 are mutually exclusive and exclude bit 8. Bits above 8 are zero.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudSMRF` status bit 1: outside the grid or not finite. */
export const GPU_POINT_CLOUD_SMRF_STATUS_INVALID = 1;
/** `GPUPointCloudSMRF` status bit 2: the point's cell has no terrain anchors. */
export const GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN = 2;
/** `GPUPointCloudSMRF` status bit 4: excluded by `selection`. */
export const GPU_POINT_CLOUD_SMRF_STATUS_EXCLUDED = 4;
/** `GPUPointCloudSMRF` status bit 8: inpainted terrain or a point near the grid edge. */
export const GPU_POINT_CLOUD_SMRF_STATUS_INPAINTED = 8;

type RasterViews = Record<
  string,
  GraphDataView<'float32'> | GraphDataView<'uint32'> | GraphDataView<'float32x3'>
>;

/**
 * Simple morphological filtering following the PDAL 2.8.4 SMRF stages: diamond openings,
 * low outliers, optional net cuts, eight-neighbor inpainting and slope-scaled classification.
 * The diamond minimum and maximum windows are the published `GPURasterErosion` and
 * `GPURasterDilation` (`pointcloud-diamond-morphology.ts`); the other stages are independently
 * implemented WGSL. No CPU numerical fallback or intermediate readback.
 */
export class GPUPointCloudSMRF implements GPUCommandNodeProducer {
  readonly id: string;
  readonly maximumRadius: number;
  readonly cutRadius: number;
  constructor(readonly props: GPUPointCloudSMRFProps) {
    this.id = props.id ?? 'gpu-pointcloud-smrf';
    validatePointCloudGrid(props.grid, this.id);
    // One-sided slope differences require at least two rows and columns.
    validateCapacity(this.id, 'grid.width', props.grid.width, 2);
    validateCapacity(this.id, 'grid.height', props.grid.height, 2);
    // Integer grid distances in the eight-neighbor fill must fit signed 32 bits.
    validateMaximumCount(
      this.id,
      'grid.width squared plus grid.height squared',
      props.grid.width ** 2 + props.grid.height ** 2,
      0x7ffffffe
    );
    this.maximumRadius = Math.ceil(
      (props.window ?? 18 * props.grid.cellSize) / props.grid.cellSize
    );
    this.cutRadius = Math.ceil((props.cut ?? 0) / props.grid.cellSize);
    validateCapacity(
      this.id,
      'window in cells',
      this.maximumRadius,
      1,
      GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS
    );
    validateCapacity(
      this.id,
      'cut in cells',
      this.cutRadius,
      0,
      GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS
    );
    validateFiniteNumber(this.id, 'slope', props.slope ?? 0.15, 0);
    validateFiniteNumber(this.id, 'scalar', props.scalar ?? 1.25, 0);
    validateFiniteNumber(this.id, 'threshold', props.threshold ?? 0.5, 0);
    validateFiniteNumber(this.id, 'cut', props.cut ?? 0, 0);
    validateCapacity(this.id, 'groundClass', props.groundClass ?? 2, 0);
    validateCapacity(this.id, 'otherClass', props.otherClass ?? 1, 0);
    validateViewLength(
      this.id,
      'groundSurface',
      props.groundSurface,
      props.grid.width * props.grid.height
    );
    validateViewLength(this.id, 'classifications', props.classifications, props.positions.length);
    validateViewLength(this.id, 'groundMask', props.groundMask, props.positions.length);
    validateViewLength(
      this.id,
      'outputClassifications',
      props.outputClassifications,
      props.positions.length
    );
    validateViewLength(this.id, 'status', props.status, props.positions.length);
    if (props.selection) {
      validateViewLength(this.id, 'selection', props.selection, props.positions.length);
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      classifications,
      selection,
      grid,
      groundSurface,
      groundMask,
      outputClassifications,
      status
    } = this.props;
    const cells = grid.width * grid.height;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let serial = 0;
    const allocate = <Format extends 'float32' | 'uint32'>(format: Format, name: string) =>
      createTransientView(graph, `${this.id}-${name}-${serial++}`, format, cells);
    const emit = (
      name: string,
      inputs: RasterViews,
      outputs: RasterViews,
      body: string,
      length = cells
    ) => {
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-${name}`,
          length,
          inputs,
          outputs,
          body
        })
      );
    };
    const indexed = createPointCloudGridIndex(
      graph,
      `${this.id}-raster-index`,
      positions,
      {
        origin: [...grid.origin, 0],
        dimensions: [grid.width, grid.height, 1],
        cellSize: grid.cellSize
      },
      true,
      getPointCloudRowDirection(grid)
    );
    nodes.push(...indexed.nodes);
    const minimum = allocate('float32', 'minimum');
    const occupied = allocate('uint32', 'occupied');
    emit(
      'minimum',
      {
        positions,
        sortedIndices: indexed.sortedIndices,
        offsets: indexed.offsets,
        ...(selection ? {selection} : {})
      },
      {minimum, occupied},
      `
var elevation = 3.402823e38; var count = 0u;
for (var slot=readOffsets(index);slot<readOffsets(index+1u);slot++) {
  let point=readSortedIndices(slot);
  ${selection ? 'if (readSelection(point) == 0u) { continue; }' : ''}
  let position = readPositions(point);
  if (!all(abs(position) <= vec3f(3.402823e38))) { continue; }
  ${getPointCloudCellSource(grid)}
  if (inside && cellIndex == index) { elevation = min(elevation,position.z); count++; }
}
writeMinimum(index,select(0.0,elevation,count>0u)); writeOccupied(index,u32(count>0u));`
    );
    const fill = (
      name: string,
      input: GraphDataView<'float32'>,
      validity: GraphDataView<'uint32'>,
      output: GraphDataView<'float32'>,
      availability: GraphDataView<'uint32'>
    ) => {
      const validCount = createTransientView(graph, `${this.id}-${name}-valid-count`, 'uint32', 1);
      // Both validity inputs hold only zero or one, so their sum is the valid cell count.
      nodes.push(
        ...new GPUReduction({
          id: `${this.id}-${name}-valid-count`,
          input: validity,
          output: validCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      emit(
        name,
        {surface: input, validity, validCount},
        {result: output, availability},
        `
if(readValidCount(0u)==0u){writeResult(index,0.0);writeAvailability(index,0u);return;}
if (readValidity(index) != 0u) { writeResult(index,readSurface(index)); writeAvailability(index,1u); return; }
var ids: array<u32,8>; var distances: array<u32,8>;
for (var slot=0u; slot<8u; slot++) { ids[slot]=0xffffffffu; distances[slot]=0xffffffffu; }
let column = i32(index % ${grid.width}u); let row = i32(index / ${grid.width}u);
for (var ring=1;ring<${Math.max(grid.width, grid.height)};ring++) {
 for (var vertical=max(-ring,-row);vertical<=min(ring,${grid.height - 1}-row);vertical++) {
  for (var horizontal=max(-ring,-column);horizontal<=min(ring,${grid.width - 1}-column);horizontal++) {
   if(max(abs(horizontal),abs(vertical))!=ring){continue;}
   let candidate=u32(row+vertical)*${grid.width}u+u32(column+horizontal);
  if (readValidity(candidate)==0u) { continue; }
  let distance = u32(horizontal*horizontal+vertical*vertical);
  let order = (candidate % ${grid.width}u)*${grid.height}u+candidate/${grid.width}u;
  var insertion = 8u;
  for (var slot=0u; slot<8u; slot++) {
    let previous = ids[slot];
    let previousOrder = (previous % ${grid.width}u)*${grid.height}u+previous/${grid.width}u;
    if (distance<distances[slot] || (distance==distances[slot] && order<previousOrder)) { insertion=slot; break; }
  }
  if (insertion<8u) {
    for (var slot=7u; slot>insertion; slot--) { ids[slot]=ids[slot-1u]; distances[slot]=distances[slot-1u]; }
    ids[insertion]=candidate; distances[insertion]=distance;
  }
  }
 }
 // Every unvisited cell is at least ring+1 away in one coordinate; strict inequality preserves cutoff ties.
 if(ids[7]!=0xffffffffu && u32((ring+1)*(ring+1))>distances[7]){break;}
}
var mean=0.0; var count=0u;
for (var slot=0u; slot<8u; slot++) { if (ids[slot]<${cells}u) { count++; mean += (readSurface(ids[slot])-mean)/f32(count); } }
writeResult(index,mean); writeAvailability(index,select(0u,2u,count>0u));`
      );
    };
    const filled = allocate('float32', 'filled');
    const filledAvailability = allocate('uint32', 'filled-availability');
    fill('initial-inpaint', minimum, occupied, filled, filledAvailability);
    const morphology = (
      name: string,
      input: GraphDataView<'float32'>,
      radius: number,
      operation: 'min' | 'max'
    ) => {
      const result = getPointCloudDiamondMorphologyNodes(
        graph,
        `${this.id}-${name}`,
        input,
        grid,
        radius,
        operation
      );
      nodes.push(...result.nodes);
      return result.output;
    };
    const inverted = allocate('float32', 'inverted');
    emit('invert', {filled}, {inverted}, 'writeInverted(index,-readFilled(index));');
    const lowOpening = morphology(
      'low-dilate',
      morphology('low-erode', inverted, 1, 'min'),
      1,
      'max'
    );
    const lowMask = allocate('uint32', 'low-mask');
    emit(
      'low-mask',
      {inverted, opening: lowOpening},
      {lowMask},
      `writeLowMask(index,u32(abs(readInverted(index)-readOpening(index))>${Math.fround(5 * grid.cellSize)}));`
    );
    let netSurface = filled;
    const netExpression = this.cutRadius
      ? `(index % ${grid.width}u) % ${this.cutRadius}u == 0u || (index / ${grid.width}u) % ${this.cutRadius}u == 0u`
      : 'false';
    if (this.cutRadius) {
      const netOpening = morphology(
        'net-dilate',
        morphology('net-erode', filled, 2 * this.cutRadius, 'min'),
        2 * this.cutRadius,
        'max'
      );
      netSurface = allocate('float32', 'net-surface');
      emit(
        'cut-net',
        {filled, opening: netOpening},
        {result: netSurface},
        `writeResult(index,select(readFilled(index),readOpening(index),${netExpression}));`
      );
    }
    let previousOpening = netSurface;
    let erosion = netSurface;
    let objectMask = allocate('uint32', 'objects');
    emit('clear-objects', {}, {objectMask}, 'writeObjectMask(index,0u);');
    for (let radius = 1; radius <= this.maximumRadius; radius++) {
      erosion = morphology(`object-erode-${radius}`, erosion, 1, 'min');
      const opening = morphology(`object-dilate-${radius}`, erosion, radius, 'max');
      const nextMask = allocate('uint32', 'objects');
      emit(
        `object-mask-${radius}`,
        {previous: previousOpening, opening, objectMask},
        {nextMask},
        `writeNextMask(index,readObjectMask(index)|u32(abs(readPrevious(index)-readOpening(index))>${Math.fround((this.props.slope ?? 0.15) * grid.cellSize * radius)}));`
      );
      objectMask = nextMask;
      previousOpening = opening;
    }
    const anchorMask = allocate('uint32', 'anchors');
    emit(
      'anchors',
      {objectMask, lowMask, availability: filledAvailability},
      {anchorMask},
      `writeAnchorMask(index,u32(readAvailability(index)>0u && readObjectMask(index)==0u && readLowMask(index)==0u && !(${netExpression})));`
    );
    const terrainAvailability = allocate('uint32', 'terrain-availability');
    fill('terrain-inpaint', filled, anchorMask, groundSurface, terrainAvailability);
    const gradients = allocate('float32', 'gradients');
    emit(
      'slope',
      {terrain: groundSurface},
      {gradients},
      `
let column = index % ${grid.width}u; let row = index / ${grid.width}u;
let left = select(column-1u,column,column==0u); let right=min(column+1u,${grid.width - 1}u);
let bottom=select(row-1u,row,row==0u); let top=min(row+1u,${grid.height - 1}u);
let horizontal=(readTerrain(row*${grid.width}u+right)-readTerrain(row*${grid.width}u+left))/(f32(right-left)*${Math.fround(grid.cellSize)});
let vertical=(readTerrain(top*${grid.width}u+column)-readTerrain(bottom*${grid.width}u+column))/(f32(top-bottom)*${Math.fround(grid.cellSize)});
writeGradients(index,length(vec2f(horizontal,vertical)));`
    );
    emit(
      'classify',
      {
        positions,
        terrain: groundSurface,
        availability: terrainAvailability,
        gradients,
        ...(selection ? {selection} : {})
      },
      {groundMask, status},
      `
writeGroundMask(index,0u);
${selection ? `if (readSelection(index)==0u) { writeStatus(index,${GPU_POINT_CLOUD_SMRF_STATUS_EXCLUDED}u); return; }` : ''}
let position=readPositions(index);
${getPointCloudCellSource(grid)}
if (!inside || !all(abs(position)<=vec3f(3.402823e38))) { writeStatus(index,${GPU_POINT_CLOUD_SMRF_STATUS_INVALID}u); return; }
if (readAvailability(cellIndex)==0u) { writeStatus(index,${GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN}u); return; }
let tolerance=${Math.fround(this.props.threshold ?? 0.5)}+${Math.fround(this.props.scalar ?? 1.25)}*readGradients(cellIndex);
writeGroundMask(index,u32(abs(position.z-readTerrain(cellIndex))<=tolerance));
let margin=${Math.max(2 * this.maximumRadius, 4 * this.cutRadius, 2)}u;
let edge = cell.x<margin || cell.y<margin || cell.x+margin>=${grid.width}u || cell.y+margin>=${grid.height}u;
writeStatus(index,select(0u,${GPU_POINT_CLOUD_SMRF_STATUS_INPAINTED}u,edge || readAvailability(cellIndex)==2u));`,
      positions.length
    );
    emit(
      'classes',
      {classifications, groundMask, status},
      {outputClassifications},
      `
let original=readClassifications(index); var result=original;
if (readGroundMask(index)!=0u) { result=${this.props.groundClass ?? 2}u; }
else if ((readStatus(index)&${GPU_POINT_CLOUD_SMRF_STATUS_INVALID | GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN | GPU_POINT_CLOUD_SMRF_STATUS_EXCLUDED}u)==0u && ${!this.props.onlyGround}) { result=${this.props.otherClass ?? 1}u; }
writeOutputClassifications(index,result);`,
      positions.length
    );
    return nodes;
  }
}
