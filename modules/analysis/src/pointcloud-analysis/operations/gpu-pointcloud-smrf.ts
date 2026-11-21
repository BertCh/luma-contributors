// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GPUReduction} from '@luma.gl/graph/gpu-core';
import {
  validateCommandNodeBudget,
  validateGraphPorts,
  validateRequirement,
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS
} from './pointcloud-limits';
import {getPointCloudDiamondMorphologyNodes} from './pointcloud-diamond-morphology';
import {
  createPointCloudGridIndex,
  getPointCloudRowDirection,
  validatePointCloudGridIndexBindings
} from './pointcloud-grid-index';
import {
  getPointCloudKernelNodes,
  validatePointCloudPackedViews,
  validatePointCloudScratchBindings
} from './pointcloud-kernel-nodes';
import {getPointCloudCellSource, validatePointCloudGrid} from './gpu-pointcloud-surfaces';
import type {PointCloudRasterGrid} from './gpu-pointcloud-surfaces';

import {
  getPointCloudRuntimeKernelNodes,
  validatePointCloudDispatchParameterBindings
} from './pointcloud-chunk-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

// Public grid sort/build (512), reductions (at most 16 per fill) and fixed raster/classify passes.
const POINT_CLOUD_SMRF_FIXED_NODE_ALLOWANCE = 640;
// Runtime mask thresholds need three common words, two settings and at most five offsets.
const POINT_CLOUD_SMRF_MAXIMUM_DISPATCH_WORD_COUNT = 10;

export type GPUPointCloudSMRFSettings = {
  /** Nonnegative terrain slope; default 0.15. */
  slope?: number;
  /** Nonnegative multiplier of the terrain gradient; default 1.25. */
  scalar?: number;
  /** Nonnegative elevation threshold in z units; default 0.5. */
  threshold?: number;
};

/** Two settings followed by one slope threshold for each compile-time radius. */
export function getGPUPointCloudSMRFParameterLength(maximumRadius: number): number {
  return 2 + maximumRadius;
}

/**
 * Pack scalar, threshold and per-radius slope thresholds for a caller-owned GPUParameterBuffer.
 * Products are evaluated in float64 and rounded once, preserving the constructor convention.
 * @param settings Per-frame classification settings.
 * @param grid The contributor's compile-time grid.
 * @param maximumRadius The contributor's compile-time maximumRadius, in cells.
 */
export function getGPUPointCloudSMRFParameterValues(
  settings: GPUPointCloudSMRFSettings,
  grid: PointCloudRasterGrid,
  maximumRadius: number
): Float32Array {
  const identifier = 'gpu-pointcloud-smrf';
  validateCapacity(
    identifier,
    'maximumRadius',
    maximumRadius,
    1,
    GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS
  );
  validateFiniteNumber(identifier, 'slope', settings.slope ?? 0.15, 0);
  validateFiniteNumber(identifier, 'scalar', settings.scalar ?? 1.25, 0);
  validateFiniteNumber(identifier, 'threshold', settings.threshold ?? 0.5, 0);
  return new Float32Array([
    settings.scalar ?? 1.25,
    settings.threshold ?? 0.5,
    ...Array.from({length: maximumRadius}, (_, index) => {
      const threshold = (settings.slope ?? 0.15) * grid.cellSize * (index + 1);
      validateFiniteNumber(identifier, `slope threshold[${index}]`, threshold, 0);
      return threshold;
    })
  ]);
}

export type GPUPointCloudSMRFOutput = {
  /** Provisional inpainted terrain in row-major cell order; zero without support. */
  groundSurface: GraphDataView<'float32'>;
  groundMask: GraphDataView<'uint32'>;
  outputClassifications: GraphDataView<'uint32'>;
  /**
   * Per-point bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_SMRF_STATUS_INVALID`): the point is
   * outside the grid or not finite. Bit 2 (`..._NO_TERRAIN`): its cell has no terrain anchors, so
   * it is not classified. Bit 4 (`..._EXCLUDED`): `mask` is zero for the point, which keeps its
   * original class. Bit 8 (`..._INPAINTED`): the terrain under the point is inpainted, or the point
   * lies within the filter margin of the grid edge, so its classification is less certain. Bits 1,
   * 2 and 4 are mutually exclusive and exclude bit 8. Bits above 8 are zero.
   */
  status: GraphDataView<'uint32'>;
};

export type GPUPointCloudSMRFProps = {
  /** Per-frame thresholds; overrides slope, scalar and threshold. */
  parameters?: GraphDataView<'float32'>;
  id?: string;
  positions: GraphDataView<'float32x3'>;
  classifications: GraphDataView<'uint32'>;
  /** Source-position-row mask: nonzero keeps a point; zero retains its original class. Scatter query-aligned filter validity through queryRows first. */
  mask?: GraphDataView<'uint32'>;
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
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudSMRFOutput;
};

/** `GPUPointCloudSMRF` status bit 1: outside the grid or not finite. */
export const GPU_POINT_CLOUD_SMRF_STATUS_INVALID = 1;
/** `GPUPointCloudSMRF` status bit 2: the point's cell has no terrain anchors. */
export const GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN = 2;
/** `GPUPointCloudSMRF` status bit 4: excluded by `mask`. */
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
 * Per-frame: input contents and parameters (slope, scalar, threshold). Compile-time: grid,
 * window, cut, class labels, onlyGround, optional mask and view shapes.
 * Without parameters, threshold props retain their constructor-time behavior.
 */
export class GPUPointCloudSMRF implements GPUCommandNodeProducer {
  readonly id: string;
  readonly maximumRadius: number;
  readonly cutRadius: number;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUPointCloudSMRFProps) {
    const identifier = props.id ?? 'gpu-pointcloud-smrf';
    validateRequirement(identifier, !('selection' in props), 'selection was removed; use mask');
    validateGraphPorts(identifier, props, {
      positions: {required: true, viewType: 'data', formats: ['float32x3']},
      classifications: {required: true, viewType: 'data', formats: ['uint32']},
      mask: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: false, viewType: 'data', formats: ['float32']},
      'output.groundSurface': {required: true, viewType: 'data', formats: ['float32']},
      'output.groundMask': {required: true, viewType: 'data', formats: ['uint32']},
      'output.outputClassifications': {required: true, viewType: 'data', formats: ['uint32']},
      'output.status': {required: true, viewType: 'data', formats: ['uint32']}
    });
    props = snapshotConstructionFields(props, [
      'parameters',
      'id',
      'positions',
      'classifications',
      'mask',
      'grid',
      'slope',
      'scalar',
      'threshold',
      'window',
      'cut',
      'groundClass',
      'otherClass',
      'onlyGround',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-smrf';
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    this.props = props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output, [
        'groundSurface',
        'groundMask',
        'outputClassifications',
        'status'
      ])
    });
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
    if (props.parameters) {
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        getGPUPointCloudSMRFParameterLength(this.maximumRadius)
      );
    }
    validateFiniteNumber(this.id, 'slope', props.slope ?? 0.15, 0);
    validateFiniteNumber(this.id, 'scalar', props.scalar ?? 1.25, 0);
    validateFiniteNumber(this.id, 'threshold', props.threshold ?? 0.5, 0);
    validateFiniteNumber(this.id, 'cut', props.cut ?? 0, 0);
    validateCapacity(this.id, 'groundClass', props.groundClass ?? 2, 0);
    validateCapacity(this.id, 'otherClass', props.otherClass ?? 1, 0);
    validateViewLength(
      this.id,
      'groundSurface',
      props.output.groundSurface,
      props.grid.width * props.grid.height
    );
    validateViewLength(this.id, 'classifications', props.classifications, props.positions.length);
    validateViewLength(this.id, 'groundMask', props.output.groundMask, props.positions.length);
    validateViewLength(
      this.id,
      'outputClassifications',
      props.output.outputClassifications,
      props.positions.length
    );
    validateViewLength(this.id, 'status', props.output.status, props.positions.length);
    if (props.mask) {
      validateViewLength(this.id, 'mask', props.mask, props.positions.length);
    }
    validateFiniteNumber(
      this.id,
      'low-outlier threshold (5 * grid.cellSize)',
      5 * props.grid.cellSize,
      0
    );
    validateFiniteNumber(
      this.id,
      'maximum slope threshold',
      (props.slope ?? 0.15) * props.grid.cellSize * this.maximumRadius,
      0
    );
    validatePointCloudPackedViews(
      this.id,
      {
        positions: props.positions,
        classifications: props.classifications,
        mask: props.mask,
        parameters: props.parameters
      },
      {
        groundSurface: props.output.groundSurface,
        groundMask: props.output.groundMask,
        outputClassifications: props.output.outputClassifications,
        status: props.output.status
      },
      [
        {
          inputs: {parameters: props.parameters},
          outputs: {groundSurface: props.output.groundSurface}
        },
        {
          inputs: {
            positions: props.positions,
            terrain: props.output.groundSurface,
            mask: props.mask,
            parameters: props.parameters
          },
          outputs: {groundMask: props.output.groundMask, status: props.output.status}
        },
        {
          inputs: {
            classifications: props.classifications,
            groundMask: props.output.groundMask,
            status: props.output.status,
            parameters: props.parameters
          },
          outputs: {outputClassifications: props.output.outputClassifications}
        }
      ]
    );
    const morphologyPasses =
      2 +
      4 * this.cutRadius +
      this.maximumRadius +
      (this.maximumRadius * (this.maximumRadius + 1)) / 2;
    this.estimatedCommandNodeCount =
      POINT_CLOUD_SMRF_FIXED_NODE_ALLOWANCE + morphologyPasses + 2 * this.maximumRadius;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      classifications,
      mask: selection,
      grid,
      parameters,
      output: {groundSurface, groundMask, outputClassifications, status}
    } = this.props;
    validateGraphChunkBindingRanges(graph, this.id, {
      positions,
      classifications,
      mask: selection,
      groundSurface,
      groundMask,
      outputClassifications,
      status,
      parameters
    });
    validatePointCloudGridIndexBindings(graph, this.id, positions, [grid.width, grid.height, 1]);
    validatePointCloudScratchBindings(graph, this.id, {
      'SMRF raster scratch and morphology validity': grid.width * grid.height * 4,
      'SMRF reduction totals': 4
    });
    validatePointCloudDispatchParameterBindings(
      graph,
      this.id,
      POINT_CLOUD_SMRF_MAXIMUM_DISPATCH_WORD_COUNT,
      1
    );
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
      length = cells,
      runtimeWords?: readonly number[]
    ) => {
      nodes.push(
        ...(runtimeWords ? getPointCloudRuntimeKernelNodes : getPointCloudKernelNodes)(graph, {
          ...(runtimeWords ? {runtimeWords} : {}),
          id: `${this.id}-${name}`,
          length,
          inputs: {...inputs, ...(parameters ? {parameters} : {})},
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
        `writeNextMask(index,readObjectMask(index)|u32(abs(readPrevious(index)-readOpening(index))>${parameters ? 'readParameters(readDispatchParameter(0u))' : 'bitcast<f32>(readDispatchParameter(1u))'}));`,
        cells,
        [
          radius + 1,
          new Uint32Array(
            new Float32Array([(this.props.slope ?? 0.15) * grid.cellSize * radius]).buffer
          )[0]
        ]
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
let tolerance=${parameters ? 'readParameters(1u)' : Math.fround(this.props.threshold ?? 0.5)}+${parameters ? 'readParameters(0u)' : Math.fround(this.props.scalar ?? 1.25)}*readGradients(cellIndex);
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
