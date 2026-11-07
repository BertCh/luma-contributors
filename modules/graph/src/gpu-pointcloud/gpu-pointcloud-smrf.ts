// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import type {GPUCommandNode} from '../gpu-core/gpu-command-node';
import {createTransientView} from '../gpu-core/graph-data-view-utils';
import {createPointCloudGridIndex} from './pointcloud-grid-index';
import {getSpatialOperationNodes} from '../utils/gpu-spatial-operation';
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
  /** 0 valid, 1 invalid/outside, 2 no terrain anchors, 4 excluded, 8 inpainted or edge-supported. */
  status: GraphDataView<'uint32'>;
};

type RasterViews = Record<
  string,
  GraphDataView<'float32'> | GraphDataView<'uint32'> | GraphDataView<'float32x3'>
>;

/**
 * Simple morphological filtering following the PDAL 2.8.4 SMRF stages: diamond openings,
 * low outliers, optional net cuts, eight-neighbor inpainting and slope-scaled classification.
 * Independently implemented WGSL; no CPU numerical fallback or intermediate readback.
 */
export class GPUPointCloudSMRF {
  readonly id: string;
  readonly maximumRadius: number;
  readonly cutRadius: number;
  constructor(readonly props: GPUPointCloudSMRFProps) {
    this.id = props.id ?? 'gpu-pointcloud-smrf';
    validatePointCloudGrid(props.grid);
    // One-sided slope differences require at least two rows and columns.
    assert(props.grid.width >= 2 && props.grid.height >= 2);
    // Integer grid distances in the eight-neighbor fill must fit signed 32 bits.
    assert(props.grid.width ** 2 + props.grid.height ** 2 < 0x7fffffff);
    this.maximumRadius = Math.ceil(
      (props.window ?? 18 * props.grid.cellSize) / props.grid.cellSize
    );
    this.cutRadius = Math.ceil((props.cut ?? 0) / props.grid.cellSize);
    assert(
      this.maximumRadius >= 1 &&
        this.maximumRadius <= 64 &&
        this.cutRadius >= 0 &&
        this.cutRadius <= 32
    );
    assert(
      [props.slope ?? 0.15, props.scalar ?? 1.25, props.threshold ?? 0.5, props.cut ?? 0].every(
        value => value >= 0 && Number.isFinite(Math.fround(value))
      )
    );
    for (const classification of [props.groundClass ?? 2, props.otherClass ?? 1])
      assert(
        Number.isInteger(classification) && classification >= 0 && classification <= 0xffffffff
      );
    assert(props.groundSurface.length === props.grid.width * props.grid.height);
    for (const view of [
      props.classifications,
      props.groundMask,
      props.outputClassifications,
      props.status
    ])
      assert(view.length === props.positions.length);
    assert(!props.selection || props.selection.length === props.positions.length);
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
        ...getSpatialOperationNodes(graph, {
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
      true
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
      emit(
        `${name}-valid-count`,
        {validity},
        {validCount},
        `var count=0u;for(var cell=0u;cell<${cells}u;cell++){count+=u32(readValidity(cell)!=0u);}writeValidCount(0u,count);`,
        1
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
      let previous = input;
      // A diamond is the Minkowski sum of radius-one crosses. Clipping to a rectangular domain
      // preserves shortest paths, so repeated crosses equal the full clipped diamond exactly.
      for (let step = 0; step < radius; step++) {
        const output = allocate('float32', `${name}-${step}`);
        emit(
          `${name}-${step}`,
          {surface: previous},
          {result: output},
          `
let column=index%${grid.width}u;let row=index/${grid.width}u;
var value=readSurface(index);
if(column>0u){value=${operation}(value,readSurface(index-1u));}
if(column+1u<${grid.width}u){value=${operation}(value,readSurface(index+1u));}
if(row>0u){value=${operation}(value,readSurface(index-${grid.width}u));}
if(row+1u<${grid.height}u){value=${operation}(value,readSurface(index+${grid.width}u));}
writeResult(index,value);`
        );
        previous = output;
      }
      return previous;
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
${selection ? 'if (readSelection(index)==0u) { writeStatus(index,4u); return; }' : ''}
let position=readPositions(index);
${getPointCloudCellSource(grid)}
if (!inside || !all(abs(position)<=vec3f(3.402823e38))) { writeStatus(index,1u); return; }
if (readAvailability(cellIndex)==0u) { writeStatus(index,2u); return; }
let tolerance=${Math.fround(this.props.threshold ?? 0.5)}+${Math.fround(this.props.scalar ?? 1.25)}*readGradients(cellIndex);
writeGroundMask(index,u32(abs(position.z-readTerrain(cellIndex))<=tolerance));
let margin=${Math.max(2 * this.maximumRadius, 4 * this.cutRadius, 2)}u;
let edge = cell.x<margin || cell.y<margin || cell.x+margin>=${grid.width}u || cell.y+margin>=${grid.height}u;
writeStatus(index,select(0u,8u,edge || readAvailability(cellIndex)==2u));`,
      positions.length
    );
    emit(
      'classes',
      {classifications, groundMask, status},
      {outputClassifications},
      `
let original=readClassifications(index); var result=original;
if (readGroundMask(index)!=0u) { result=${this.props.groundClass ?? 2}u; }
else if ((readStatus(index)&7u)==0u && ${!this.props.onlyGround}) { result=${this.props.otherClass ?? 1}u; }
writeOutputClassifications(index,result);`,
      positions.length
    );
    return nodes;
  }
}
