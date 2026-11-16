// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
  GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
} from './pointcloud-limits';
import {
  createPointCloudGridIndex,
  getPointCloudRowDirection,
  validatePointCloudGridIndexBindings
} from './pointcloud-grid-index';
import {getPointCloudKernelNodes, validatePointCloudPackedViews} from './pointcloud-kernel-nodes';

import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../analysis-core/kernels/graph-chunk-dispatch';
import {
  getPointCloudChunkKernelNodes,
  validatePointCloudChunkViews
} from './pointcloud-chunk-kernel-nodes';
import type {PointCloudKernelProps} from './pointcloud-kernel-nodes';

import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';

// One exact uint32 classification, read each encoding rather than specialized into the pipeline.
export const GPU_POINT_CLOUD_SURFACES_PARAMETER_LENGTH = 1;
// Exact coverage mask/support/loop reach and bitcast float32 radius squared/power divided by two.
export const GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH = 5;

export type GPUPointCloudSurfacesSettings = {groundClass?: number};

/** Pack the per-frame ground classification. */
export function getGPUPointCloudSurfacesParameterValues(
  settings: GPUPointCloudSurfacesSettings = {}
): Uint32Array {
  validateCapacity('gpu-pointcloud-surfaces', 'groundClass', settings.groundClass ?? 2, 0);
  return new Uint32Array([settings.groundClass ?? 2]);
}

export type GPUPointCloudHeightSettings = {
  coverageBit?: number;
  interpolation?: GPUPointCloudHeightProps['interpolation'];
};

/** Pack runtime sampling settings; interpolation method and grid stay compile-time. */
export function getGPUPointCloudHeightParameterValues(
  settings: GPUPointCloudHeightSettings,
  grid: PointCloudRasterGrid
): Uint32Array {
  const identifier = 'gpu-pointcloud-height';
  validatePointCloudGrid(grid, identifier);
  validateCapacity(identifier, 'coverageBit', settings.coverageBit ?? 0xffffffff);
  const interpolation = settings.interpolation;
  const values = new Uint32Array(GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH);
  values[0] = settings.coverageBit ?? 0xffffffff;
  if (interpolation?.method === 'idw') {
    validatePointCloudHeightInterpolation(identifier, interpolation, grid);
    new Float32Array(values.buffer).set(
      [interpolation.radius ** 2, (interpolation.power ?? 2) / 2],
      1
    );
    values[3] = interpolation.minimumSupport ?? 1;
    values[4] = Math.ceil(interpolation.radius / grid.cellSize) + 1;
  }
  return values;
}

function validatePointCloudHeightInterpolation(
  identifier: string,
  interpolation: Extract<NonNullable<GPUPointCloudHeightProps['interpolation']>, {method: 'idw'}>,
  grid: PointCloudRasterGrid
): void {
  validateRequirement(
    identifier,
    interpolation.radius > 0 &&
      Math.fround(interpolation.radius ** 2) > 0 &&
      Number.isFinite(Math.fround(interpolation.radius ** 2)),
    'interpolation.radius must be positive with a finite nonzero float32 square'
  );
  validateMaximumCount(
    identifier,
    'interpolation.radius in cells',
    interpolation.radius / grid.cellSize,
    GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
  );
  validateRequirement(
    identifier,
    Math.fround((interpolation.power ?? 2) / 2) > 0 &&
      (interpolation.power ?? 2) <= GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
    `interpolation.power must be positive in float32 and at most GPU_POINT_CLOUD_MAXIMUM_IDW_POWER (${GPU_POINT_CLOUD_MAXIMUM_IDW_POWER})`
  );
  validateCapacity(identifier, 'interpolation.minimumSupport', interpolation.minimumSupport ?? 1);
}

/**
 * Raster grid of the point-cloud contributors. `origin` is the lower-left corner (minimum x and y)
 * and column `c`, row `r` is cell `r * width + c`.
 */
export type PointCloudRasterGrid = {
  origin: readonly [number, number];
  cellSize: number;
  width: number;
  height: number;
  /**
   * Which way row indices run. `'south'` (default): row 0 is the top row at the maximum y and rows
   * run toward decreasing y, the convention of the raster and terrain families (GDAL and image
   * order, and the one `GPURasterMetadata` and the D8 flow codes assume), so these rasters feed
   * `GPUTerrainFlow`, the derivatives or the illumination contributors without flipping rows.
   * `'north'`: row 0 is the bottom row at the minimum y and rows run toward increasing y.
   *
   * Unlike the terrain classes, this grid has no affine or signed cell height, so the direction
   * cannot be read from it and no contradiction can be detected: every contributor that takes the
   * same grid must be given the same direction.
   */
  rowDirection?: 'north' | 'south';
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
  validateCapacity(id, 'grid.width', grid.width, 1, GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS);
  validateCapacity(id, 'grid.height', grid.height, 1, GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS);
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
  validateRequirement(
    id,
    grid.rowDirection === undefined ||
      grid.rowDirection === 'north' ||
      grid.rowDirection === 'south',
    "grid.rowDirection must be 'north' or 'south'"
  );
}

/** @internal */
export function getPointCloudCellSource(
  grid: PointCloudRasterGrid,
  point = 'position',
  runtimeGrid = false
): string {
  const origin = runtimeGrid
    ? 'vec2f(readGridParameters(0u),readGridParameters(1u))'
    : `vec2f(${grid.origin.map(Math.fround).join(',')})`;
  const cellSize = runtimeGrid ? 'readGridParameters(3u)' : `${Math.fround(grid.cellSize)}`;
  // `coordinate` is in cell units with y in row units: for 'south' rows count down from the top edge.
  // Membership is decided on the offset from the origin, exactly as in the north case, so a point
  // on a cell boundary belongs to the cell above it (half-open) in both directions: the south row
  // is height - 1 - floor(offset.y) in integers. `height - offset.y` is only the continuous row
  // coordinate for interpolation; it rounds in float32 (up, when offset.y is just below a small
  // integer), so it must not decide the row or the edges.
  if (getPointCloudRowDirection(grid) === 'south') {
    return `let southOffset = getPointCloudQuotient2(${point}.xy - ${origin}, ${cellSize});
let coordinate = vec2f(southOffset.x, ${grid.height}.0 - southOffset.y);
let inside = all(southOffset >= vec2f(0.0)) && all(southOffset < vec2f(${grid.width}.0, ${grid.height}.0));
let northCell = vec2u(clamp(floor(southOffset), vec2f(0.0), vec2f(${grid.width - 1}.0, ${grid.height - 1}.0)));
let cell = vec2u(northCell.x, ${grid.height - 1}u - northCell.y);
let cellIndex = cell.y * ${grid.width}u + cell.x;`;
  }
  return `let coordinate = getPointCloudQuotient2(${point}.xy - ${origin}, ${cellSize});
let inside = all(coordinate >= vec2f(0.0)) && all(coordinate < vec2f(${grid.width}.0, ${grid.height}.0));
let cell = vec2u(clamp(floor(coordinate), vec2f(0.0), vec2f(${grid.width - 1}.0, ${grid.height - 1}.0)));
let cellIndex = cell.y * ${grid.width}u + cell.x;`;
}

export type GPUPointCloudSurfacesProps = {
  /** Runtime origin/cellSize, packed by getGPUPointCloudGridParameterValues; grid dimensions stay fixed. */
  gridParameters?: GraphDataView<'float32'>;
  /** Runtime groundClass, packed by getGPUPointCloudSurfacesParameterValues. */
  parameters?: GraphDataView<'uint32'>;
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

/**
 * Explicit cell-min DTM and cell-max DSM, with no interpolation or implicit datum conversion.
 * Per-frame: input contents, parameters (groundClass), and gridParameters (origin/cellSize).
 * Compile-time: grid dimensions/row direction and view shapes.
 * Without parameters, groundClass retains its constructor-time behavior.
 */
export class GPUPointCloudSurfaces implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudSurfacesProps) {
    this.id = props.id ?? 'gpu-pointcloud-surfaces';
    validatePointCloudGrid(props.grid, this.id);
    validatePointCloudGridParameters(this.id, props.gridParameters);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['classifications', props.classifications, 'uint32'],
      ['terrain', props.terrain, 'float32'],
      ['surface', props.surface, 'float32'],
      ['canopy', props.canopy, 'float32'],
      ['coverage', props.coverage, 'uint32'],
      ['parameters', props.parameters, 'uint32']
    ] as const)
      if (view) validateSinglePackedView(view, [format], `${this.id} ${name}`);
    if (props.parameters)
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_SURFACES_PARAMETER_LENGTH
      );
    validateViewLength(this.id, 'classifications', props.classifications, props.positions.length);
    validateCapacity(this.id, 'groundClass', props.groundClass ?? 2, 0);
    const cells = props.grid.width * props.grid.height;
    validateViewLength(this.id, 'terrain', props.terrain, cells);
    validateViewLength(this.id, 'surface', props.surface, cells);
    validateViewLength(this.id, 'canopy', props.canopy, cells);
    validateViewLength(this.id, 'coverage', props.coverage, cells);
    const inputs = {
      positions: props.positions,
      classifications: props.classifications,
      parameters: props.parameters,
      gridParameters: props.gridParameters
    };
    const outputs = {
      terrain: props.terrain,
      surface: props.surface,
      canopy: props.canopy,
      coverage: props.coverage
    };
    validatePointCloudPackedViews(
      this.id,
      inputs,
      outputs,
      props.gridParameters
        ? [{inputs, outputs: {terrain: props.terrain, surface: props.surface}}, {outputs}]
        : props.parameters
          ? [
              {
                inputs,
                outputs: {terrain: props.terrain, surface: props.surface, coverage: props.coverage}
              },
              {
                inputs: {terrain: props.terrain, surface: props.surface, coverage: props.coverage},
                outputs: {canopy: props.canopy}
              }
            ]
          : [{inputs, outputs}]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {positions, classifications, grid, terrain, surface, canopy, coverage} = this.props;
    validateGraphChunkBindingRanges(graph, this.id, {
      positions,
      classifications,
      terrain,
      surface,
      canopy,
      coverage,
      parameters: this.props.parameters,
      gridParameters: this.props.gridParameters
    });
    validatePointCloudGridIndexBindings(
      graph,
      this.id,
      positions,
      [grid.width, grid.height, 1],
      this.props.gridParameters
    );
    const indexed = createPointCloudGridIndex(
      graph,
      `${this.id}-index`,
      positions,
      {
        origin: [...grid.origin, 0],
        dimensions: [grid.width, grid.height, 1],
        cellSize: grid.cellSize
      },
      true,
      getPointCloudRowDirection(grid),
      this.props.gridParameters
    );
    return [
      ...indexed.nodes,
      ...getPointCloudKernelNodes(graph, {
        id: this.id,
        length: terrain.length,
        inputs: {
          positions,
          classifications,
          ...(this.props.parameters ? {parameters: this.props.parameters} : {}),
          ...(this.props.gridParameters ? {gridParameters: this.props.gridParameters} : {}),
          sortedIndices: indexed.sortedIndices,
          offsets: indexed.offsets
        },
        outputs: {
          terrain,
          surface,
          ...(this.props.gridParameters
            ? {}
            : {coverage, ...(this.props.parameters ? {} : {canopy})})
        },
        body: `var minimum = 3.402823e38; var maximum = -3.402823e38; var flags = 0u;
for (var slot = readOffsets(index); slot < readOffsets(index+1u); slot++) {
  let point = readSortedIndices(slot);
  let position = readPositions(point);
  if (!all(abs(position) <= vec3f(3.402823e38))) { continue; }
  ${getPointCloudCellSource(grid, 'position', Boolean(this.props.gridParameters))}
  if (!inside || cellIndex != index) { continue; }
  flags |= ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT}u; maximum = max(maximum, position.z);
  if (readClassifications(point) == ${this.props.parameters ? 'readParameters(0u)' : `${this.props.groundClass ?? 2}u`}) { minimum = min(minimum, position.z); flags |= ${GPU_POINT_CLOUD_COVERAGE_GROUND}u; }
}
writeTerrain(index, select(${this.props.gridParameters ? 'bitcast<f32>(0x7fc00000u | (index & 0u))' : '0.0'}, minimum, (flags & ${GPU_POINT_CLOUD_COVERAGE_GROUND}u) != 0u));
writeSurface(index, select(${this.props.gridParameters ? 'bitcast<f32>(0x7fc00000u | (index & 0u))' : '0.0'}, maximum, (flags & ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT}u) != 0u));
${this.props.parameters || this.props.gridParameters ? '' : `writeCanopy(index, select(0.0, maximum - minimum, flags == ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT | GPU_POINT_CLOUD_COVERAGE_GROUND}u));`}
${this.props.gridParameters ? '' : 'writeCoverage(index, flags);'}`
      }),
      ...(this.props.parameters && !this.props.gridParameters
        ? getPointCloudKernelNodes(graph, {
            id: `${this.id}-canopy`,
            length: terrain.length,
            inputs: {terrain, surface, coverage},
            outputs: {canopy},
            body: `writeCanopy(index, select(0.0, readSurface(index) - readTerrain(index), readCoverage(index) == ${GPU_POINT_CLOUD_COVERAGE_ANY_POINT | GPU_POINT_CLOUD_COVERAGE_GROUND}u));`
          })
        : []),
      ...(this.props.gridParameters
        ? getPointCloudKernelNodes(graph, {
            id: `${this.id}-finalize`,
            length: terrain.length,
            inputs: {},
            outputs: {terrain, surface, canopy, coverage},
            body: `let minimum = readTerrain(index); let maximum = readSurface(index);
let hasGround = abs(minimum) <= 3.402823e38; let hasPoint = abs(maximum) <= 3.402823e38;
writeCoverage(index, u32(hasPoint) | (u32(hasGround) * 2u));
writeTerrain(index, select(0.0, minimum, hasGround)); writeSurface(index, select(0.0, maximum, hasPoint));
writeCanopy(index, select(0.0, maximum - minimum, hasGround && hasPoint));`
          })
        : [])
    ];
  }
}

export type GPUPointCloudHeightProps = {
  /** Runtime origin/cellSize, packed by getGPUPointCloudGridParameterValues; grid dimensions stay fixed. */
  gridParameters?: GraphDataView<'float32'>;
  /** Runtime coverageBit and IDW numeric settings, packed by getGPUPointCloudHeightParameterValues. */
  parameters?: GraphDataView<'uint32'>;
  id?: string;
  positions: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  grid: PointCloudRasterGrid;
  terrain: GraphDataView<'float32'>;
  /** Nonzero cells are supported. For Surfaces coverage pass coverageBit: 2. */
  coverage: GraphDataView<'uint32'>;
  coverageBit?: number;
  /** Cell sampling by default; bilinear requires every positive-weight corner, IDW fills from finite supported centers inside radius. */
  interpolation?:
    | {method: 'bilinear'}
    | {method: 'idw'; radius: number; power?: number; minimumSupport?: number};
  heights: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * Per-point status code (a value, not a bit set): `GPU_POINT_CLOUD_HEIGHT_STATUS_VALID` (0)
   * valid sample, `..._INVALID_POINT` (1) the point is outside the grid or not finite,
   * `..._UNSUPPORTED_TERRAIN` (2) the terrain under the point is unsupported (or the interpolation
   * lacks support), `..._NONFINITE` (3) the elevation or height overflowed float32.
   */
  status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/** `GPUPointCloudHeight` status code: valid sample. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_VALID = 0;
/** `GPUPointCloudHeight` status code: the point is outside the grid or has a nonfinite coordinate. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT = 1;
/** `GPUPointCloudHeight` status code: unsupported terrain under the point, or too little interpolation support. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN = 2;
/** `GPUPointCloudHeight` status code: the elevation or height is not finite in float32. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE = 3;

/**
 * Height above ground, preserving negative heights. Positions, heights and status accept identical
 * ordered chunk lengths; terrain/coverage stay resident. Per-frame: parameters, gridParameters
 * (origin/cellSize) and input contents. Compile-time: grid dimensions/row direction, interpolation
 * method and chunk topology. IDW reach remains bounded by
 * GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS when packing parameters.
 */
export class GPUPointCloudHeight implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  constructor(readonly props: GPUPointCloudHeightProps) {
    this.id = props.id ?? 'gpu-pointcloud-height';
    validatePointCloudGrid(props.grid, this.id);
    validatePointCloudGridParameters(this.id, props.gridParameters);
    if (props.interpolation?.method === 'idw')
      validatePointCloudHeightInterpolation(this.id, props.interpolation, props.grid);
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['uint32'], `${this.id} parameters`);
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH
      );
    }
    validateSinglePackedView(props.terrain, ['float32'], `${this.id} terrain`);
    validateSinglePackedView(props.coverage, ['uint32'], `${this.id} coverage`);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['heights', props.heights, 'float32'],
      ['status', props.status, 'uint32']
    ] as const)
      for (const chunk of getGraphViewChunks(view))
        validateSinglePackedView(chunk, [format], `${this.id} ${name}`);
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      positions: props.positions,
      heights: props.heights,
      status: props.status
    });
    validatePointCloudChunkViews(
      this.id,
      this.chunkPlan,
      ['positions'],
      ['heights', 'status'],
      {
        terrain: props.terrain,
        coverage: props.coverage,
        parameters: props.parameters,
        gridParameters: props.gridParameters
      },
      [
        {
          inputNames: ['positions'],
          outputNames: ['heights', 'status'],
          inputs: {
            terrain: props.terrain,
            coverage: props.coverage,
            parameters: props.parameters,
            gridParameters: props.gridParameters
          }
        }
      ]
    );
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
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
    if (!this.chunkPlan.nonemptyChunkCount) return [];
    validateGraphChunkDispatchBindings(graph, this.id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, this.id, {
      terrain,
      coverage,
      parameters: this.props.parameters,
      gridParameters: this.props.gridParameters
    });
    const coverageMask = this.props.parameters
      ? 'readParameters(0u)'
      : `${this.props.coverageBit ?? 0xffffffff}u`;
    const supported = `(readCoverage(sampleIndex)&${coverageMask})!=0u && abs(readTerrain(sampleIndex))<=3.402823e38`;
    let sample = `if ((readCoverage(cellIndex) & ${coverageMask}) == 0u || !(abs(readTerrain(cellIndex)) <= 3.402823e38)) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u); return; }
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
      const reach = this.props.gridParameters
        ? '(i32(ceil(getScaledQuotient(sqrt(radiusSquared), readGridParameters(3u), 1.0, 0))) + 1)'
        : this.props.parameters
          ? 'i32(readParameters(4u))'
          : Math.ceil(interpolation.radius / grid.cellSize) + 1;
      const radiusSquared = this.props.parameters
        ? 'bitcast<f32>(readParameters(1u))'
        : Math.fround(interpolation.radius ** 2);
      const halfPower = this.props.parameters
        ? 'bitcast<f32>(readParameters(2u))'
        : Math.fround((interpolation.power ?? 2) / 2);
      const minimumSupport = this.props.parameters
        ? 'readParameters(3u)'
        : `${interpolation.minimumSupport ?? 1}u`;
      sample = `${
        this.props.gridParameters
          ? `let radiusSquared = ${radiusSquared};
if (!(getScaledQuotient(sqrt(radiusSquared), readGridParameters(3u), 1.0, 0) <= ${GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS}.0)) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u); return; }`
          : ''
      }
let center=vec2i(floor(coordinate));var meanElevation=0.0;var totalWeight=0.0;var minimumSquaredDistance=3.402823e38;var support=0u;var exact=false;
for(var row=max(0,center.y-${reach});row<=min(${grid.height - 1},center.y+${reach});row++){
 for(var column=max(0,center.x-${reach});column<=min(${grid.width - 1},center.x+${reach});column++){
  let sampleIndex=u32(row)*${grid.width}u+u32(column);if(!(${supported})){continue;}
  let difference=(vec2f(f32(column),f32(row))+vec2f(0.5)-coordinate)*${this.props.gridParameters ? 'readGridParameters(3u)' : Math.fround(grid.cellSize)};
  let distanceSquared=dot(difference,difference);if(distanceSquared>${radiusSquared}){continue;}
  support++;if(distanceSquared==0.0){exact=true;elevation=readTerrain(sampleIndex);continue;}
  // Normalize to the closest sample so near-centre queries cannot overflow inverse powers.
  if(distanceSquared<minimumSquaredDistance){
   totalWeight*=pow(getScaledQuotient(distanceSquared,minimumSquaredDistance,1.0,0),${halfPower});
   minimumSquaredDistance=distanceSquared;
  }
  let weight=pow(getScaledQuotient(minimumSquaredDistance,distanceSquared,1.0,0),${halfPower});
  let combinedWeight=totalWeight+weight;
  meanElevation=meanElevation*(totalWeight/combinedWeight)+readTerrain(sampleIndex)*(weight/combinedWeight);
  totalWeight=combinedWeight;
 }}
let valid=exact||(support>=${minimumSupport}&&totalWeight>0.0);
if(!exact&&totalWeight>0.0){elevation=meanElevation;}if(!valid){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u);return;}`;
    }
    const kernelProps: PointCloudKernelProps = {
      id: this.id,
      length: positions.length,
      inputs: {
        positions: getGraphViewChunks(positions)[0],
        terrain,
        coverage,
        ...(this.props.parameters ? {parameters: this.props.parameters} : {}),
        ...(this.props.gridParameters ? {gridParameters: this.props.gridParameters} : {})
      },
      outputs: {heights: getGraphViewChunks(heights)[0], status: getGraphViewChunks(status)[0]},
      body: `let position = readPositions(index);
${getPointCloudCellSource(grid, 'position', Boolean(this.props.gridParameters))}
writeHeights(index, 0.0);
if (!inside || !all(abs(position) <= vec3f(3.402823e38))) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT}u); return; }
var elevation=0.0;
${sample}
let height=position.z-elevation;
// Finite observations can still overflow interpolation or the final signed difference.
if(any((bitcast<vec2u>(vec2f(elevation,height))&vec2u(0x7f800000u))==vec2u(0x7f800000u))){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE}u);return;}
writeHeights(index,height);writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_VALID}u);`
    };
    return getPointCloudChunkKernelNodes(graph, {...kernelProps, plan: this.chunkPlan});
  }
}
