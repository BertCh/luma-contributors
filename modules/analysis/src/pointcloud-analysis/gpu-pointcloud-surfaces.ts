// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createPointCloudGridIndex} from './pointcloud-grid-index';
import {getSpatialOperationNodes} from '@luma.gl/graph';

export type PointCloudRasterGrid = {
  origin: readonly [number, number];
  cellSize: number;
  width: number;
  height: number;
};

/** @internal */
export function validatePointCloudGrid(grid: PointCloudRasterGrid): void {
  assert(
    Number.isInteger(grid.width) &&
      grid.width > 0 &&
      Number.isInteger(grid.height) &&
      grid.height > 0
  );
  assert(grid.width * grid.height < 0xffffffff);
  assert(Math.fround(grid.cellSize) > 0 && Number.isFinite(Math.fround(grid.cellSize)));
  assert(grid.origin.every(value => Number.isFinite(Math.fround(value))));
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
  /** Bit 1 has any point; bit 2 has classified ground. */
  coverage: GraphDataView<'uint32'>;
};

/** Explicit cell-min DTM and cell-max DSM, with no interpolation or implicit datum conversion. */
export class GPUPointCloudSurfaces {
  readonly id: string;
  constructor(readonly props: GPUPointCloudSurfacesProps) {
    this.id = props.id ?? 'gpu-pointcloud-surfaces';
    validatePointCloudGrid(props.grid);
    assert(props.classifications.length === props.positions.length);
    assert(
      Number.isInteger(props.groundClass ?? 2) &&
        (props.groundClass ?? 2) >= 0 &&
        (props.groundClass ?? 2) <= 0xffffffff
    );
    for (const view of [props.terrain, props.surface, props.canopy, props.coverage])
      assert(view.length === props.grid.width * props.grid.height);
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
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
      ...getSpatialOperationNodes(graph, {
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
  flags |= 1u; maximum = max(maximum, position.z);
  if (readClassifications(point) == ${this.props.groundClass ?? 2}u) { minimum = min(minimum, position.z); flags |= 2u; }
}
writeTerrain(index, select(0.0, minimum, (flags & 2u) != 0u));
writeSurface(index, select(0.0, maximum, (flags & 1u) != 0u));
writeCanopy(index, select(0.0, maximum - minimum, flags == 3u));
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
  /** 0 valid sample, 1 outside grid/invalid point, 2 unsupported terrain, 3 nonfinite elevation/height. */
  status: GraphDataView<'uint32'>;
};

/** Height above ground using the containing terrain cell, explicitly preserving negative heights. */
export class GPUPointCloudHeight {
  readonly id: string;
  constructor(readonly props: GPUPointCloudHeightProps) {
    this.id = props.id ?? 'gpu-pointcloud-height';
    if (props.interpolation?.method === 'idw') {
      assert(
        props.interpolation.radius > 0 &&
          Math.fround(props.interpolation.radius ** 2) > 0 &&
          Number.isFinite(Math.fround(props.interpolation.radius ** 2))
      );
      assert(props.interpolation.radius / props.grid.cellSize <= 64);
      assert((props.interpolation.power ?? 2) > 0 && (props.interpolation.power ?? 2) <= 8);
      assert(
        Number.isSafeInteger(props.interpolation.minimumSupport ?? 1) &&
          (props.interpolation.minimumSupport ?? 1) > 0 &&
          (props.interpolation.minimumSupport ?? 1) <= 0xffffffff
      );
    }
    validatePointCloudGrid(props.grid);
    assert(
      props.terrain.length === props.grid.width * props.grid.height &&
        props.coverage.length === props.terrain.length
    );
    assert(
      props.heights.length === props.positions.length &&
        props.status.length === props.positions.length
    );
    assert(
      Number.isInteger(props.coverageBit ?? 0xffffffff) &&
        (props.coverageBit ?? 0xffffffff) > 0 &&
        (props.coverageBit ?? 0xffffffff) <= 0xffffffff
    );
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {positions, grid, terrain, coverage, heights, status, interpolation} = this.props;
    const supported = `(readCoverage(sampleIndex)&${this.props.coverageBit ?? 0xffffffff}u)!=0u && abs(readTerrain(sampleIndex))<=3.402823e38`;
    let sample = `if ((readCoverage(cellIndex) & ${this.props.coverageBit ?? 0xffffffff}u) == 0u || !(abs(readTerrain(cellIndex)) <= 3.402823e38)) { writeStatus(index, 2u); return; }
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
if(!valid){writeStatus(index,2u);return;}`;
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
if(!exact&&totalWeight>0.0){elevation=meanElevation;}if(!valid){writeStatus(index,2u);return;}`;
    }
    return getSpatialOperationNodes(graph, {
      id: this.id,
      length: positions.length,
      inputs: {positions, terrain, coverage},
      outputs: {heights, status},
      body: `let position = readPositions(index);
${getPointCloudCellSource(grid)}
writeHeights(index, 0.0);
if (!inside || !all(abs(position) <= vec3f(3.402823e38))) { writeStatus(index, 1u); return; }
var elevation=0.0;
${sample}
let height=position.z-elevation;
// Finite observations can still overflow interpolation or the final signed difference.
if(any((bitcast<vec2u>(vec2f(elevation,height))&vec2u(0x7f800000u))==vec2u(0x7f800000u))){writeStatus(index,3u);return;}
writeHeights(index,height);writeStatus(index,0u);`
    });
  }
}
