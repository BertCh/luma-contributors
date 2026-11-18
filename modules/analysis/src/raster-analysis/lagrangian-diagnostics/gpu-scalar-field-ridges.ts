// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getRasterGroundCellSizeWGSL,
  writeRasterGridParameterValues,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SMALLEST_SYMMETRIC_EIGEN_2_WGSL} from '../../analysis-core/kernels/symmetric-eigen-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBand,
  validateRasterBuffersDistinct,
  validateRasterBandBelongsToGraph,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterGridOrthogonal
} from '../raster-band-utils';

const OPERATION = 'GPUScalarFieldRidges';
/** Maximum grid axis: bounds row-major addressing to uint32. */
export const GPU_SCALAR_FIELD_RIDGES_MAXIMUM_GRID_AXIS = 65535;
/** Number of float32 parameters: grid scales/edges, thresholds and two reserved words. */
export const GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH = 8;

/** Per-frame ridge thresholds. */
export type GPUScalarFieldRidgesSettings = {
  /** Lowest accepted calibrated field value. Defaults to 0. */
  minimumValue?: number;
  /** Lowest accepted downward curvature, in field units per squared distance. Defaults to 0; flat cells are always rejected. */
  minimumCurvature?: number;
};

/**
 * Packs grid cell sizes and row edges plus the ridge thresholds.
 *
 * Layout: `[cellSizeX, cellSizeY, northEdge, southEdge, minimumValue, minimumCurvature, 0, 0]`.
 *
 * @param settings Per-frame thresholds.
 * @param grid Grid affine; update these parameters when the affine changes.
 * @param target Optional destination of at least 8 float32 elements.
 */
export function getGPUScalarFieldRidgesParameterValues(
  settings: GPUScalarFieldRidgesSettings,
  grid: GPURasterMetadata,
  target = new Float32Array(GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH)
): Float32Array {
  validateRasterGrid(OPERATION, grid);
  validateRasterGridOrthogonal(OPERATION, grid);
  validateRequirement(
    OPERATION,
    target.length >= GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH,
    `target must hold at least ${GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH} elements`
  );
  const minimumValue = settings.minimumValue ?? 0;
  const minimumCurvature = settings.minimumCurvature ?? 0;
  validateFiniteNumber(OPERATION, 'minimumValue', minimumValue);
  validateFiniteNumber(OPERATION, 'minimumCurvature', minimumCurvature, 0);
  validateRequirement(
    OPERATION,
    Math.hypot(grid.affine[0], grid.affine[3]) > 0 &&
      Math.hypot(grid.affine[1], grid.affine[4]) > 0,
    'grid.affine pixel axis lengths must be > 0'
  );
  writeRasterGridParameterValues(target, grid, {cellSizeIndex: 0, northEdgeIndex: 2});
  target.set([minimumValue, minimumCurvature, 0, 0], 4);
  return target;
}

/** Fixed-sized outputs: one row per input cell, with no compaction or capacity truncation. */
export type GPUScalarFieldRidgesOutput = {
  /** 1 for a detected ridge cell, otherwise 0. Border and invalid stencils are 0. */
  ridges: GraphDataView<'uint32'>;
  /** 1 for a finite, fully valid interior 3 by 3 stencil, otherwise 0. */
  validity: GraphDataView<'uint32'>;
  /** Optional max(0, -smallest Hessian eigenvalue) for every valid stencil; 0 where invalid. */
  strength?: GraphDataView<'float32'>;
};

/** Properties for {@link GPUScalarFieldRidges}. */
export type GPUScalarFieldRidgesProps = {
  id?: string;
  /** Scalar raster: utilization density, cumulative utilization volume, or FTLE. Calibration and validity are honoured. */
  input: GPURasterBand;
  /** Orthogonal pixel axes; rotated uniform grids are supported, sheared grids are rejected. */
  grid: GPURasterMetadata;
  /** Planar by default; longitude-latitude requires cellSizeMode 'geographic'. */
  spatialContext?: GPUSpatialContext;
  /** Ground-distance conversion at each stencil's center row; defaults to 'uniform'. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Per-frame grid and thresholds from {@link getGPUScalarFieldRidgesParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: GPUScalarFieldRidgesOutput;
};

/**
 * Raster approximation of one-dimensional height ridges of a two-dimensional scalar field.
 * Central differences over a 3 by 3 stencil give its gradient and Hessian. Spacing quotients
 * combine integer exponents without forming spacing products. A rationalized 2 by 2 eigenpair
 * calculation scales only its direction calculation, retaining diagonals in their original units.
 * The most negative
 * Hessian eigenvalue must be strictly negative and meet minimumCurvature. The normal Newton step
 * `-dot(gradient, normal) / eigenvalue` must lie inside the cell's half-open pixel box, and the
 * center value must meet minimumValue. Ties in the eigenvalues choose the column axis; normals
 * have positive column component (or positive row component when the column component is zero).
 * The half-open box [-0.5, 0.5) assigns an exact half-cell tie to the larger column/row.
 *
 * This is a local, subcell ridge detector, not connected-curve tracing or a proof of an LCS.
 * Constant plateaus and valleys are rejected. Borders, invalid band samples and nonfinite
 * stencils yield ridges=0, validity=0, strength=0. Grid dimensions below 3 are rejected.
 *
 * Per-frame: band contents, parameters (including grid affine scales and edges).
 * Compile-time: grid dimensions, spatialContext, cellSizeMode, band metadata and optional output.
 * Geographic and Web Mercator distances use a row-local WGS84 tangent scale, not a covariant
 * Hessian on the sphere. Uniform rotated grids use the orthogonal pixel basis.
 *
 * Reference: Peikert and Sadlo (2008), Height Ridge Computation and Filtering for Visualization,
 * https://cgl.ethz.ch/Downloads/Publications/Papers/2008/Pei08/Pei08.pdf .
 */
export class GPUScalarFieldRidges implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUScalarFieldRidgesProps;

  constructor(props: GPUScalarFieldRidgesProps) {
    const id = props.id ?? 'scalar-field-ridges';
    const {grid, input, output} = props;
    validateCapacity(id, 'grid.width', grid.width, 3, GPU_SCALAR_FIELD_RIDGES_MAXIMUM_GRID_AXIS);
    validateCapacity(id, 'grid.height', grid.height, 3, GPU_SCALAR_FIELD_RIDGES_MAXIMUM_GRID_AXIS);
    const count = validateRasterGrid(id, grid);
    validateRasterGridOrthogonal(id, grid);
    validateRasterGridAxisAligned(id, grid, props.cellSizeMode);
    validateRequirement(
      id,
      Math.hypot(grid.affine[0], grid.affine[3]) > 0 &&
        Math.hypot(grid.affine[1], grid.affine[4]) > 0,
      'grid.affine pixel axis lengths must be > 0'
    );
    validateRequirement(
      id,
      ['uniform', 'geographic', 'web-mercator'].includes(props.cellSizeMode ?? 'uniform'),
      'cellSizeMode must be uniform, geographic or web-mercator'
    );
    const spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      coordinateSpaces: ['planar', 'longitude-latitude'],
      metricsByCoordinateSpace: {planar: ['none', 'native'], 'longitude-latitude': ['none']}
    });
    validateRequirement(
      id,
      (spatialContext.coordinateSpace === 'longitude-latitude') ===
        (props.cellSizeMode === 'geographic'),
      'longitude-latitude spatialContext and geographic cellSizeMode must be supplied together'
    );
    validateRasterBand(input, grid, `${id} input`);
    for (const view of getRasterBandViews(input)) {
      validateSinglePackedView(
        view,
        [view === input.validity ? 'uint32' : input.format],
        `${id} input`
      );
      validateViewLength(id, 'input', view, count);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH,
      `parameters must contain at least ${GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH} float32 rows`
    );
    for (const [name, view, format] of [
      ['output.ridges', output.ridges, 'uint32'],
      ['output.validity', output.validity, 'uint32'],
      ['output.strength', output.strength, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} ${name}`);
        validateViewLength(id, name, view, count);
      }
    }
    validateRasterBuffersDistinct(
      id,
      [output.ridges, output.validity, output.strength],
      [...getRasterBandViews(input), props.parameters]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, input, output} = props;
    validateRasterBandBelongsToGraph(id, graph, input, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.ridges,
      output.validity,
      output.strength
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'grid cells',
      grid.width * grid.height * 4,
      'use a smaller grid'
    );
    const source = getRasterBufferBandNodes(graph, id, input, grid, 'reuse-canonical');
    return [
      ...source.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ridges`,
        operation: OPERATION,
        invocationCount: grid.width * grid.height,
        bindings: [
          {name: 'values', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'inputValidity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'ridges', view: output.ridges, type: 'u32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          ...(output.strength
            ? [
                {
                  name: 'strength',
                  view: output.strength,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        declarations: `${IS_FINITE_VALUE_WGSL}
${SMALLEST_SYMMETRIC_EIGEN_2_WGSL}
${SCALED_QUOTIENT_WGSL}
const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;
${getRasterGroundCellSizeWGSL(props.cellSizeMode ?? 'uniform', {settingsName: 'parameters', cellSizeIndex: 0, northEdgeIndex: 2})}
fn readValue(column: u32, row: u32) -> f32 { return values[valuesOffset + row * WIDTH + column]; }`,
        body: /* wgsl */ `
  let column = index % WIDTH;
  let row = index / WIDTH;
  ridges[ridgesOffset + index] = 0u;
  validity[validityOffset + index] = 0u;
  ${output.strength ? 'strength[strengthOffset + index] = 0.0;' : ''}
  if (column == 0u || row == 0u || column + 1u == WIDTH || row + 1u == HEIGHT) { return; }
  for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
    for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
      let neighbor = u32(i32(row) + rowOffset) * WIDTH + u32(i32(column) + columnOffset);
      if (inputValidity[inputValidityOffset + neighbor] == 0u || !isFiniteValue(values[valuesOffset + neighbor])) { return; }
    }
  }
  let spacing = getGroundCellSize(row);
  if (!all(spacing > vec2<f32>(0.0)) || !isFiniteValue(spacing.x) || !isFiniteValue(spacing.y)) { return; }
  let center = readValue(column, row);
  let west = readValue(column - 1u, row);
  let east = readValue(column + 1u, row);
  let previousRow = readValue(column, row - 1u);
  let nextRow = readValue(column, row + 1u);
  let gradient = vec2<f32>(getScaledQuotient(east - west, spacing.x, 1.0, -1),
    getScaledQuotient(nextRow - previousRow, spacing.y, 1.0, -1));
  let columnCurvature = getScaledQuotient((east - center) + (west - center), spacing.x, spacing.x, 0);
  let rowCurvature = getScaledQuotient((nextRow - center) + (previousRow - center), spacing.y, spacing.y, 0);
  let crossCurvature = getScaledQuotient((readValue(column + 1u, row + 1u) - readValue(column - 1u, row + 1u)) -
    (readValue(column + 1u, row - 1u) - readValue(column - 1u, row - 1u)), spacing.x, spacing.y, -2);
  if (!isFiniteValue(columnCurvature) || !isFiniteValue(rowCurvature) ||
      !isFiniteValue(crossCurvature) || !isFiniteValue(gradient.x) || !isFiniteValue(gradient.y)) { return; }
  // Keep the smaller diagonal in its original units: a common Hessian scale can underflow
  // it even when the final eigenvalue is normal. Only the bounded eigenvector tangent is scaled.
  let eigensystem = getSmallestSymmetricEigen2(columnCurvature, crossCurvature, rowCurvature);
  let smallest = eigensystem.value;
  if (!isFiniteValue(smallest)) { return; }
  validity[validityOffset + index] = 1u;
  ${output.strength ? 'strength[strengthOffset + index] = max(0.0, -smallest);' : ''}
  if (!(smallest < 0.0) || -smallest < parameters[parametersOffset + 5u] || center < parameters[parametersOffset + 4u]) { return; }
  var normal = eigensystem.vector;
  if (normal.x < 0.0 || (normal.x == 0.0 && normal.y < 0.0)) { normal = -normal; }
  let displacement = -dot(gradient, normal) / smallest;
  let pixelOffset = displacement * normal / spacing;
  if (all(pixelOffset >= vec2<f32>(-0.5)) && all(pixelOffset < vec2<f32>(0.5))) {
    ridges[ridgesOffset + index] = 1u;
  }`
      })
    ];
  }
}
