// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS
} from './pointcloud-limits';
import {
  createPointCloudGridIndex,
  getPointCloudRowDirection,
  validatePointCloudGridIndexBindings
} from './pointcloud-grid-index';
import {getPointCloudKernelNodes, validatePointCloudPackedViews} from './pointcloud-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';

// One exact uint32 classification, read each encoding rather than specialized into the pipeline.
export const GPU_POINT_CLOUD_SURFACES_PARAMETER_LENGTH = 1;

export type GPUPointCloudSurfacesSettings = {groundClass?: number};

/** Pack the per-frame ground classification. */
export function getGPUPointCloudSurfacesParameterValues(
  settings: GPUPointCloudSurfacesSettings = {}
): Uint32Array {
  validateCapacity('gpu-pointcloud-surfaces', 'groundClass', settings.groundClass ?? 2, 0);
  return new Uint32Array([settings.groundClass ?? 2]);
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

export type GPUPointCloudSurfacesOutput = {
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
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudSurfacesOutput;
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
    props = snapshotConstructionFields(props, [
      'gridParameters',
      'parameters',
      'id',
      'positions',
      'classifications',
      'groundClass',
      'grid',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-surfaces';
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const grid = snapshotConstructionFields(props.grid, [
      'origin',
      'cellSize',
      'width',
      'height',
      'rowDirection'
    ]);
    this.props = props = Object.freeze({
      ...props,
      grid: Object.freeze({
        ...grid,
        origin: Object.freeze([grid.origin[0], grid.origin[1]] as const)
      }),
      output: snapshotConstructionFields(props.output, ['terrain', 'surface', 'canopy', 'coverage'])
    });
    validatePointCloudGrid(props.grid, this.id);
    validatePointCloudGridParameters(this.id, props.gridParameters);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['classifications', props.classifications, 'uint32'],
      ['terrain', props.output.terrain, 'float32'],
      ['surface', props.output.surface, 'float32'],
      ['canopy', props.output.canopy, 'float32'],
      ['coverage', props.output.coverage, 'uint32'],
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
    validateViewLength(this.id, 'terrain', props.output.terrain, cells);
    validateViewLength(this.id, 'surface', props.output.surface, cells);
    validateViewLength(this.id, 'canopy', props.output.canopy, cells);
    validateViewLength(this.id, 'coverage', props.output.coverage, cells);
    const inputs = {
      positions: props.positions,
      classifications: props.classifications,
      parameters: props.parameters,
      gridParameters: props.gridParameters
    };
    const outputs = {
      terrain: props.output.terrain,
      surface: props.output.surface,
      canopy: props.output.canopy,
      coverage: props.output.coverage
    };
    validatePointCloudPackedViews(
      this.id,
      inputs,
      outputs,
      props.gridParameters
        ? [
            {inputs, outputs: {terrain: props.output.terrain, surface: props.output.surface}},
            {outputs}
          ]
        : props.parameters
          ? [
              {
                inputs,
                outputs: {
                  terrain: props.output.terrain,
                  surface: props.output.surface,
                  coverage: props.output.coverage
                }
              },
              {
                inputs: {
                  terrain: props.output.terrain,
                  surface: props.output.surface,
                  coverage: props.output.coverage
                },
                outputs: {canopy: props.output.canopy}
              }
            ]
          : [{inputs, outputs}]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      classifications,
      grid,
      output: {terrain, surface, canopy, coverage}
    } = this.props;
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
