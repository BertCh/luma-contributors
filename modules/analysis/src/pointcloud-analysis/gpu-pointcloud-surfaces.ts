// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
  GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
} from './pointcloud-limits';
import {createPointCloudGridIndex} from './pointcloud-grid-index';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';

export type PointCloudRasterGrid = {
  origin: readonly [number, number];
  cellSize: number;
  width: number;
  height: number;
};

/**
 * Throws unless a raster grid has positive integer dimensions below `GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS`
 * cells, a positive float32 cell size and a finite origin.
 *
 * @param grid Grid to check.
 * @param id Contributor ID, for the message.
 * @internal
 */
export function validatePointCloudGrid(grid: PointCloudRasterGrid, id = 'pointcloud'): void {
  validateCapacity(id, 'grid.width', grid.width);
  validateCapacity(id, 'grid.height', grid.height);
  validateMaximumCount(
    id,
    'grid cell count',
    grid.width * grid.height,
    GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS
  );
  validateFiniteNumber(id, 'grid.cellSize', grid.cellSize);
  validateRequirement(
    id,
    Math.fround(grid.cellSize) > 0,
    'grid.cellSize must be positive in float32'
  );
  validateRequirement(
    id,
    grid.origin.every(value => Number.isFinite(Math.fround(value))),
    'grid.origin must hold finite numbers'
  );
}

/** @internal */
export function getPointCloudCellSource(grid: PointCloudRasterGrid, point = 'position'): string {
  return `let coordinate = (${point}.xy - vec2f(${grid.origin.map(Math.fround).join(',')})) / ${Math.fround(grid.cellSize)};
let inside = all(coordinate >= vec2f(0.0)) && all(coordinate < vec2f(${grid.width}.0, ${grid.height}.0));
let cell = vec2u(clamp(floor(coordinate), vec2f(0.0), vec2f(${grid.width - 1}.0, ${grid.height - 1}.0)));
let cellIndex = cell.y * ${grid.width}u + cell.x;`;
}

export type GPUPointCloudSurfacesProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  classifications: GraphDataView<'uint32'>;
  groundClass?: number;
  grid: PointCloudRasterGrid;
  /** Minimum classified-ground elevation per cell; zero if not covered. */
  terrain: GraphDataView<'float32'>;
  /** Maximum elevation of all finite points per cell. */
  surface: GraphDataView<'float32'>;
  /** Surface minus terrain; negative values are preserved. */
  canopy: GraphDataView<'float32'>;
  /**
   * Per-cell coverage bits: `GPU_POINT_CLOUD_COVERAGE_ANY_POINT` (1) the cell holds a finite point,
   * `GPU_POINT_CLOUD_COVERAGE_GROUND` (2) it holds a point of the ground class. Other bits are zero.
   */
  coverage: GraphDataView<'uint32'>;
};

/** `GPUPointCloudSurfaces` coverage bit: the cell holds at least one finite point. */
export const GPU_POINT_CLOUD_COVERAGE_ANY_POINT = 1;
/** `GPUPointCloudSurfaces` coverage bit: the cell holds at least one point of the ground class. */
export const GPU_POINT_CLOUD_COVERAGE_GROUND = 2;

/** Explicit cell-min DTM and cell-max DSM, with no interpolation or implicit datum conversion. */
export class GPUPointCloudSurfaces implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudSurfacesProps) {
    this.id = props.id ?? 'gpu-pointcloud-surfaces';
    validatePointCloudGrid(props.grid, this.id);
    validateViewLength(this.id, 'classifications', props.classifications, props.positions.length);
    validateCapacity(this.id, 'groundClass', props.groundClass ?? 2, 0);
    const cells = props.grid.width * props.grid.height;
    validateViewLength(this.id, 'terrain', props.terrain, cells);
    validateViewLength(this.id, 'surface', props.surface, cells);
    validateViewLength(this.id, 'canopy', props.canopy, cells);
    validateViewLength(this.id, 'coverage', props.coverage, cells);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {positions, classifications, grid, terrain, surface, canopy, coverage} = this.props;
    const indexed = createPointCloudGridIndex(
      graph,
      `${this.id}-index`,
      positions,
      {
        origin: [...grid.origin, 0],
        dimensions: [grid.width, grid.height, 1],
        cellSize: grid.cellSize
      },
      true
    );
    return [
      ...indexed.nodes,
      ...getPointCloudKernelNodes(graph, {
        id: this.id,
        length: terrain.length,
        inputs: {
          positions,
          classifications,
          sortedIndices: indexed.sortedIndices,
          offsets: indexed.offsets
        },
        outputs: {terrain, surface, canopy, coverage},
        body: `var minimum = 3.402823e38; var maximum = -3.402823e38; var flags = 0u;
for (var slot = readOffsets(index); slot < readOffsets(index+1u); slot++) {
  let point = readSortedIndices(slot);
  let position = readPositions(point);
  if (!all(abs(position) <= vec3f(3.402823e38))) { continue; }
  ${getPointCloudCellSource(grid)}
  if (!inside || cellIndex != index) { continue; }
  flags |= ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT}u; maximum = max(maximum, position.z);
  if (readClassifications(point) == ${this.props.groundClass ?? 2}u) { minimum = min(minimum, position.z); flags |= ${GPU_POINT_CLOUD_COVERAGE_GROUND}u; }
}
writeTerrain(index, select(0.0, minimum, (flags & ${GPU_POINT_CLOUD_COVERAGE_GROUND}u) != 0u));
writeSurface(index, select(0.0, maximum, (flags & ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT}u) != 0u));
writeCanopy(index, select(0.0, maximum - minimum, flags == ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT | GPU_POINT_CLOUD_COVERAGE_GROUND}u));
writeCoverage(index, flags);`
      })
    ];
  }
}

export type GPUPointCloudHeightProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  grid: PointCloudRasterGrid;
  terrain: GraphDataView<'float32'>;
  /** Nonzero cells are supported. For Surfaces coverage pass coverageBit: 2. */
  coverage: GraphDataView<'uint32'>;
  coverageBit?: number;
  /** Cell sampling by default; bilinear requires every positive-weight corner, IDW fills from finite supported centers inside radius. */
  interpolation?:
    | {method: 'bilinear'}
    | {method: 'idw'; radius: number; power?: number; minimumSupport?: number};
  heights: GraphDataView<'float32'>;
  /**
   * Per-point status code (a value, not a bit set): `GPU_POINT_CLOUD_HEIGHT_STATUS_VALID` (0)
   * valid sample, `..._INVALID_POINT` (1) the point is outside the grid or not finite,
   * `..._UNSUPPORTED_TERRAIN` (2) the terrain under the point is unsupported (or the interpolation
   * lacks support), `..._NONFINITE` (3) the elevation or height overflowed float32.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudHeight` status code: valid sample. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_VALID = 0;
/** `GPUPointCloudHeight` status code: the point is outside the grid or has a nonfinite coordinate. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT = 1;
/** `GPUPointCloudHeight` status code: unsupported terrain under the point, or too little interpolation support. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN = 2;
/** `GPUPointCloudHeight` status code: the elevation or height is not finite in float32. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE = 3;

/** Height above ground using the containing terrain cell, explicitly preserving negative heights. */
export class GPUPointCloudHeight implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudHeightProps) {
    this.id = props.id ?? 'gpu-pointcloud-height';
    validatePointCloudGrid(props.grid, this.id);
    if (props.interpolation?.method === 'idw') {
      validateRequirement(
        this.id,
        props.interpolation.radius > 0 &&
          Math.fround(props.interpolation.radius ** 2) > 0 &&
          Number.isFinite(Math.fround(props.interpolation.radius ** 2)),
        'interpolation.radius must be positive with a finite nonzero float32 square'
      );
      validateMaximumCount(
        this.id,
        'interpolation.radius in cells',
        props.interpolation.radius / props.grid.cellSize,
        GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
      );
      validateRequirement(
        this.id,
        (props.interpolation.power ?? 2) > 0 &&
          (props.interpolation.power ?? 2) <= GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
        `interpolation.power must be in (0, ${GPU_POINT_CLOUD_MAXIMUM_IDW_POWER}]`
      );
      validateCapacity(
        this.id,
        'interpolation.minimumSupport',
        props.interpolation.minimumSupport ?? 1
      );
    }
    const cells = props.grid.width * props.grid.height;
    validateViewLength(this.id, 'terrain', props.terrain, cells);
    validateViewLength(this.id, 'coverage', props.coverage, cells);
    validateViewLength(this.id, 'heights', props.heights, props.positions.length);
    validateViewLength(this.id, 'status', props.status, props.positions.length);
    validateCapacity(this.id, 'coverageBit', props.coverageBit ?? 0xffffffff);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {positions, grid, terrain, coverage, heights, status, interpolation} = this.props;
    const supported = `(readCoverage(sampleIndex)&${this.props.coverageBit ?? 0xffffffff}u)!=0u && abs(readTerrain(sampleIndex))<=3.402823e38`;
    let sample = `if ((readCoverage(cellIndex) & ${this.props.coverageBit ?? 0xffffffff}u) == 0u || !(abs(readTerrain(cellIndex)) <= 3.402823e38)) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u); return; }
elevation = readTerrain(cellIndex);`;
    if (interpolation?.method === 'bilinear')
      sample = `
let centered=coordinate-vec2f(0.5);let lower=vec2i(floor(centered));let fraction=fract(centered);
var valid=true;
for(var vertical=0;vertical<2;vertical++){for(var horizontal=0;horizontal<2;horizontal++){
 let weight=select(1.0-fraction.x,fraction.x,horizontal==1)*select(1.0-fraction.y,fraction.y,vertical==1);
 if(weight==0.0){continue;}let location=lower+vec2i(horizontal,vertical);
 if(any(location<vec2i(0))||any(location>=vec2i(${grid.width},${grid.height}))){valid=false;continue;}
 let sampleIndex=u32(location.y)*${grid.width}u+u32(location.x);
 if(!(${supported})){valid=false;continue;}elevation+=weight*readTerrain(sampleIndex);
}}
if(!valid){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u);return;}`;
    if (interpolation?.method === 'idw') {
      const reach = Math.ceil(interpolation.radius / grid.cellSize) + 1;
      sample = `
let center=vec2i(floor(coordinate));var meanElevation=0.0;var totalWeight=0.0;var minimumSquaredDistance=3.402823e38;var support=0u;var exact=false;
for(var row=max(0,center.y-${reach});row<=min(${grid.height - 1},center.y+${reach});row++){
 for(var column=max(0,center.x-${reach});column<=min(${grid.width - 1},center.x+${reach});column++){
  let sampleIndex=u32(row)*${grid.width}u+u32(column);if(!(${supported})){continue;}
  let difference=(vec2f(f32(column),f32(row))+vec2f(0.5)-coordinate)*${Math.fround(grid.cellSize)};
  let distanceSquared=dot(difference,difference);if(distanceSquared>${Math.fround(interpolation.radius ** 2)}){continue;}
  support++;if(distanceSquared==0.0){exact=true;elevation=readTerrain(sampleIndex);continue;}
  // Normalize to the closest sample so near-centre queries cannot overflow inverse powers.
  if(distanceSquared<minimumSquaredDistance){
   totalWeight*=pow(distanceSquared/minimumSquaredDistance,${Math.fround((interpolation.power ?? 2) / 2)});
   minimumSquaredDistance=distanceSquared;
  }
  let weight=pow(minimumSquaredDistance/distanceSquared,${Math.fround((interpolation.power ?? 2) / 2)});
  let combinedWeight=totalWeight+weight;
  meanElevation=meanElevation*(totalWeight/combinedWeight)+readTerrain(sampleIndex)*(weight/combinedWeight);
  totalWeight=combinedWeight;
 }}
let valid=exact||(support>=${interpolation.minimumSupport ?? 1}u&&totalWeight>0.0);
if(!exact&&totalWeight>0.0){elevation=meanElevation;}if(!valid){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u);return;}`;
    }
    return getPointCloudKernelNodes(graph, {
      id: this.id,
      length: positions.length,
      inputs: {positions, terrain, coverage},
      outputs: {heights, status},
      body: `let position = readPositions(index);
${getPointCloudCellSource(grid)}
writeHeights(index, 0.0);
if (!inside || !all(abs(position) <= vec3f(3.402823e38))) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT}u); return; }
var elevation=0.0;
${sample}
let height=position.z-elevation;
// Finite observations can still overflow interpolation or the final signed difference.
if(any((bitcast<vec2u>(vec2f(elevation,height))&vec2u(0x7f800000u))==vec2u(0x7f800000u))){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE}u);return;}
writeHeights(index,height);writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_VALID}u);`
    });
  }
}
