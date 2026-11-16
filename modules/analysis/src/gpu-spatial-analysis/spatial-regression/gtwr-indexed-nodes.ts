// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';
import {BIN_INDEX_WGSL} from '../shared/bin-index-wgsl';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  getEquilibratedCholeskyWGSL,
  getLocalKernelWeightWGSL
} from './local-weighted-least-squares-wgsl';
import {
  getGTWRGridAdaptiveBandwidthWGSL,
  GPU_GTWR_MINIMUM_CANDIDATE_RADIUS,
  GPU_GTWR_MINIMUM_NORMAL_VALUE
} from './gtwr-adaptive-grid-wgsl';
import {createGTWRIndexedSummaryNodes} from './gtwr-indexed-summary';
import type {GPUGeographicallyTemporallyWeightedRegressionProps} from './gpu-geographically-temporally-weighted-regression';

/** Target occupancy controls grid memory; no cell or neighborhood is truncated. */
const GPU_GTWR_GRID_ROWS_PER_CELL = 8;
/** Packed records add x, y, time and response to the predictors (fit records use the same stride). */
const GPU_GTWR_RECORD_EXTRA_WORDS = 4;

/** A cubic grid over the data extent; timeScale zero collapses assignments to the first time slice. @internal */
export function getGTWRGridDimension(rows: number): number {
  return Math.max(1, Math.ceil(Math.cbrt(rows / GPU_GTWR_GRID_ROWS_PER_CELL)));
}

/** Preflights every derived binding before graph resources are declared. @internal */
export function validateGTWRIndexedBindings(
  device: Device,
  id: string,
  rows: number,
  predictorCount: number
): void {
  const dimension = getGTWRGridDimension(rows);
  const boundsPartialCount = Math.min(256, Math.ceil(rows / 4096));
  const sumPartialCount = Math.min(1024, Math.ceil(rows / 256));
  for (const [name, byteLength] of [
    ['packed input and fit records', rows * (predictorCount + GPU_GTWR_RECORD_EXTRA_WORDS) * 4],
    ['grid cell offsets', (dimension ** 3 + 1) * 4],
    ['grid keys and source rows', rows * 4],
    ['grid radix histogram and scan scratch', Math.ceil(rows / 256) * 16 * 4],
    ['bounds', 16],
    ['bounds partials', boundsPartialCount > 1 ? boundsPartialCount * 16 : 0],
    ['summary partials', sumPartialCount * 4 * 4]
  ] as const) {
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      byteLength,
      'reduce rows or predictorCount'
    );
  }
}

/** Stable 3D cell candidates and parallel local fits for fixed or adaptive compact support. @internal */
export function createGTWRIndexedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUGeographicallyTemporallyWeightedRegressionProps
): GPUCommandNode<Parameters>[] {
  const rows = props.response.length;
  const columns = props.predictorCount;
  const size = columns + 1;
  const stride = columns + GPU_GTWR_RECORD_EXTRA_WORDS;
  const dimension = getGTWRGridDimension(rows);
  const cellCount = dimension ** 3;
  const operation = 'GPUGeographicallyTemporallyWeightedRegression';
  const records = createTransientView(graph, `${id}-records`, 'float32', rows * stride);
  const fits = createTransientView(graph, `${id}-fits`, 'float32', rows * stride);
  const spatialBounds = createTransientView(graph, `${id}-spatial-bounds`, 'float32', 4);
  const temporalBounds = createTransientView(graph, `${id}-temporal-bounds`, 'float32', 4);
  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rows);
  const sourceRows = createTransientView(graph, `${id}-source-rows`, 'uint32', rows);
  const common = `const RECORD_WORDS: u32 = ${stride}u;
const GRID_DIMENSION: u32 = ${dimension}u;
${IS_FINITE_BITS_WGSL}
${GET_NAN_WGSL}
${BIN_INDEX_WGSL}
// Opposite finite extrema can overflow their difference; scale only the bin mapping in that case.
fn getGTWRBinIndex(value: f32, lower: f32, upper: f32) -> u32 {
  if (!isFiniteBits(upper - lower)) {
    return getBinIndex(value * 0.5, lower * 0.5, upper * 0.5, GRID_DIMENSION);
  }
  return getBinIndex(value, lower, upper, GRID_DIMENSION);
}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-pack-records`,
      operation,
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'times', view: props.times, type: 'f32', access: 'read'},
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'records', view: records, type: 'f32', access: 'read_write'}
      ],
      declarations: common,
      body: `let base = recordsOffset + index * RECORD_WORDS;
  let positionX = positions[positionsOffset + 2u * index];
  let positionY = positions[positionsOffset + 2u * index + 1u];
  let time = times[timesOffset + index];
  let observed = response[responseOffset + index];
  var included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  included = included && isFiniteBits(positionX) && isFiniteBits(positionY) && isFiniteBits(time) && isFiniteBits(observed);
  for (var column = 0u; column < ${columns}u; column++) {
    let predictor = predictors[predictorsOffset + index * ${columns}u + column];
    included = included && isFiniteBits(predictor);
    records[base + 4u + column] = predictor;
  }
  records[base] = select(getNaN(), positionX, included);
  records[base + 1u] = positionY;
  records[base + 2u] = time;
  records[base + 3u] = observed;`
    })
  ];
  for (const [name, output, firstComponent, secondComponent] of [
    ['spatial', spatialBounds, 0, 1],
    ['temporal', temporalBounds, 2, 2]
  ] as const) {
    nodes.push(
      ...getBoundsReductionNodes(graph, {
        id: `${id}-${name}`,
        operation,
        variant: `${name}-extent`,
        itemCount: rows,
        output,
        bindings: [{name: 'records', view: records, type: 'f32', access: 'read'}],
        declarations: `${common}
fn readItem(row: u32) -> vec4f {
  let base = recordsOffset + row * RECORD_WORDS;
  let included = isFiniteBits(records[base]);
  return vec4f(records[base + ${firstComponent}u], records[base + ${secondComponent}u], select(0.0, 1.0, included), 0.0);
}`
      })
    );
  }
  // Integer keys avoid a second float normalization when the published index is built. The
  // shared helper composes published group aggregation, scan and stable radix sort, so every
  // cell retains ascending source-row order, including coincident observations.
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-cell-keys`,
      operation,
      invocationCount: rows,
      bindings: [
        {name: 'records', view: records, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'spatialBounds', view: spatialBounds, type: 'f32', access: 'read'},
        {name: 'temporalBounds', view: temporalBounds, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'sourceRows', view: sourceRows, type: 'u32', access: 'read_write'}
      ],
      declarations: common,
      body: `let base = recordsOffset + index * RECORD_WORDS;
  var key = ${cellCount}u;
  if (isFiniteBits(records[base])) {
    let column = getGTWRBinIndex(records[base], spatialBounds[spatialBoundsOffset], spatialBounds[spatialBoundsOffset + 2u]);
    let gridRow = getGTWRBinIndex(records[base + 1u], spatialBounds[spatialBoundsOffset + 1u], spatialBounds[spatialBoundsOffset + 3u]);
    var timeCell = 0u;
    if (parameters[parametersOffset + 1u] > 0.0) {
      timeCell = getGTWRBinIndex(records[base + 2u], temporalBounds[temporalBoundsOffset], temporalBounds[temporalBoundsOffset + 2u]);
    }
    key = (timeCell * GRID_DIMENSION + gridRow) * GRID_DIMENSION + column;
  }
  cellKeys[cellKeysOffset + index] = key;
  sourceRows[sourceRowsOffset + index] = index;`
    })
  );
  const grid = createPointNeighborGridIndexNodes(graph, {
    id: `${id}-grid`,
    rowCount: rows,
    cellKeys,
    sourceRows,
    cellCount
  });
  nodes.push(...grid.nodes);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-fit`,
      operation,
      variant: 'indexed-local-fit',
      invocationCount: rows,
      bindings: [
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'records', view: records, type: 'f32', access: 'read'},
        {name: 'cellOffsets', view: grid.cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: grid.sortedRows, type: 'u32', access: 'read'},
        {name: 'spatialBounds', view: spatialBounds, type: 'f32', access: 'read'},
        {name: 'temporalBounds', view: temporalBounds, type: 'f32', access: 'read'},
        {name: 'fits', view: fits, type: 'f32', access: 'read_write'}
      ],
      declarations: `${common}
${IS_FINITE_VALUE_WGSL}
${getCholeskyWGSL(size, size)}
${getEquilibratedCholeskyWGSL(size)}
${getLocalKernelWeightWGSL(props.gaussianCutoff ?? null)}
${props.adaptive ? getGTWRGridAdaptiveBandwidthWGSL() : ''}
// Inclusive padded cell range. Exact original metric/weight tests reject false positives.
fn getCellRange(origin: f32, reach: f32, lower: f32, upper: f32) -> vec2u {
  if (!isFiniteBits(reach)) { return vec2u(0u, GRID_DIMENSION - 1u); }
  let padded = reach * 1.0001 + abs(origin) * 2.4e-7;
  return vec2u(getGTWRBinIndex(clamp(origin - padded, lower, upper), lower, upper),
    getGTWRBinIndex(clamp(origin + padded, lower, upper), lower, upper));
}`,
      body: getIndexedFitBody(
        columns,
        props.kernel === 'bisquare',
        props.gaussianCutoff !== undefined,
        props.adaptive ?? false
      )
    })
  );
  const {output} = props;
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-publish-fit`,
      operation,
      invocationCount: rows,
      bindings: [
        {name: 'fits', view: fits, type: 'f32', access: 'read'},
        {name: 'coefficients', view: output.coefficients, type: 'f32', access: 'read_write'},
        {name: 'fitted', view: output.fitted, type: 'f32', access: 'read_write'},
        {name: 'hatDiagonal', view: output.hatDiagonal, type: 'f32', access: 'read_write'},
        {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read_write'}
      ],
      body: `let base = fitsOffset + index * ${stride}u;
  for (var column = 0u; column < ${size}u; column++) {
    coefficients[coefficientsOffset + index * ${size}u + column] = fits[base + column];
  }
  fitted[fittedOffset + index] = fits[base + ${size}u];
  hatDiagonal[hatDiagonalOffset + index] = fits[base + ${size + 1}u];
  localStatus[localStatusOffset + index] = u32(fits[base + ${size + 2}u]);`
    })
  );
  if (output.residuals) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-residuals`,
        operation,
        invocationCount: rows,
        bindings: [
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'fitted', view: output.fitted, type: 'f32', access: 'read'},
          {name: 'residuals', view: output.residuals, type: 'f32', access: 'read_write'}
        ],
        body: 'residuals[residualsOffset + index] = response[responseOffset + index] - fitted[fittedOffset + index];'
      })
    );
  }
  nodes.push(...createGTWRIndexedSummaryNodes(graph, id, props));
  return nodes;
}

/** Same centered normal equations and Cholesky solve as the dense reference. */
function getIndexedFitBody(
  columns: number,
  bisquare: boolean,
  hasCutoff: boolean,
  adaptive: boolean
): string {
  const size = columns + 1;
  return `let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${adaptive ? 'getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)' : 'parameters[parametersOffset]'};
  let originX = records[recordsOffset + RECORD_WORDS * index];
  var code = 1u;
  var coefficient: array<f32, ${size}>;
  for (var column = 0u; column < ${size}u; column++) {
    coefficient[column] = getNaN();
  }
  var fittedValue = getNaN();
  var hat = getNaN();
  if (!isFiniteBits(originX)) {
    code = 2u;
  } else if (bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale)) {
    let originY = records[recordsOffset + RECORD_WORDS * index + 1u];
    let originTime = records[recordsOffset + RECORD_WORDS * index + 2u];
    let originBase = recordsOffset + index * RECORD_WORDS + 4u;
    var normalMatrix: array<f32, ${size * size}>;
    var normalResponse: array<f32, ${size}>;
    var design: array<f32, ${size}>;
    for (var entry = 0u; entry < ${size * size}u; entry++) { normalMatrix[entry] = 0.0; }
    for (var column = 0u; column < ${size}u; column++) { normalResponse[column] = 0.0; }
    design[0] = 1.0;
    let radius = max(${bisquare || !hasCutoff ? 'bandwidth' : 'bandwidth * GAUSSIAN_CUTOFF'}, ${GPU_GTWR_MINIMUM_CANDIDATE_RADIUS});
    let columnRange = getCellRange(originX, radius, spatialBounds[spatialBoundsOffset], spatialBounds[spatialBoundsOffset + 2u]);
    let rowRange = getCellRange(originY, radius, spatialBounds[spatialBoundsOffset + 1u], spatialBounds[spatialBoundsOffset + 3u]);
    var timeRange = vec2u(0u);
    if (timeScale > 0.0) {
      // Account for either multiplication order of tau * dt * dt and subnormal flushing.
      let timeReach = max(max(radius / sqrt(timeScale), ${GPU_GTWR_MINIMUM_CANDIDATE_RADIUS}), ${GPU_GTWR_MINIMUM_NORMAL_VALUE} / timeScale);
      timeRange = getCellRange(originTime, timeReach, temporalBounds[temporalBoundsOffset], temporalBounds[temporalBoundsOffset + 2u]);
    }
    for (var timeCell = timeRange.x; timeCell <= timeRange.y; timeCell++) {
      for (var gridRow = rowRange.x; gridRow <= rowRange.y; gridRow++) {
        let cellBase = (timeCell * GRID_DIMENSION + gridRow) * GRID_DIMENSION;
        let firstSlot = cellOffsets[cellOffsetsOffset + cellBase + columnRange.x];
        let endSlot = cellOffsets[cellOffsetsOffset + cellBase + columnRange.y + 1u];
        for (var slot = firstSlot; slot < endSlot; slot++) {
          let row = sortedRows[sortedRowsOffset + slot];
          let rowX = records[recordsOffset + RECORD_WORDS * row];
          if (!isFiniteBits(rowX)) { continue; }
          let deltaX = rowX - originX;
          let deltaY = records[recordsOffset + RECORD_WORDS * row + 1u] - originY;
          let deltaTime = records[recordsOffset + RECORD_WORDS * row + 2u] - originTime;
          let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
          let weight = getLocalKernelWeight(distance, bandwidth, ${bisquare ? 'true' : 'false'});
          if (!(weight > 0.0)) { continue; }
          let rowBase = recordsOffset + row * RECORD_WORDS + 4u;
          for (var column = 0u; column < ${columns}u; column++) {
            design[column + 1u] = records[rowBase + column] - records[originBase + column];
          }
          let observed = records[recordsOffset + row * RECORD_WORDS + 3u];
          for (var firstColumn = 0u; firstColumn < ${size}u; firstColumn++) {
            let weighted = weight * design[firstColumn];
            normalResponse[firstColumn] = normalResponse[firstColumn] + weighted * observed;
            for (var secondColumn = 0u; secondColumn <= firstColumn; secondColumn++) {
              normalMatrix[firstColumn * ${size}u + secondColumn] = normalMatrix[firstColumn * ${size}u + secondColumn] + weighted * design[secondColumn];
            }
          }
        }
      }
    }
    var scale: array<f32, ${size}>;
    if (factorEquilibrated_${size}(&normalMatrix, &scale)) {
      let inverse00 = inverseDiagonalEquilibrated_${size}(&normalMatrix, &scale, 0u);
      // normalResponse is X'Wy of the centered design; the solve is in place and unscales the coefficients.
      solveEquilibrated_${size}(&normalMatrix, &scale, &normalResponse);
      var finite = isFiniteBits(inverse00) && inverse00 >= 0.0;
      for (var column = 0u; column < ${size}u; column++) {
        finite = finite && isFiniteBits(normalResponse[column]);
      }
      if (finite) {
        code = 0u;
        // The centered intercept is the fitted value; uncentering it gives the raw intercept.
        fittedValue = normalResponse[0];
        hat = inverse00;
        var intercept = normalResponse[0];
        for (var column = 0u; column < ${columns}u; column++) {
          coefficient[column + 1u] = normalResponse[column + 1u];
          intercept = intercept - normalResponse[column + 1u] * records[originBase + column];
        }
        coefficient[0] = intercept;
      }
    }
  }
  for (var column = 0u; column < ${size}u; column++) {
    fits[fitsOffset + index * RECORD_WORDS + column] = coefficient[column];
  }
  fits[fitsOffset + index * RECORD_WORDS + ${size}u] = fittedValue;
  fits[fitsOffset + index * RECORD_WORDS + ${size + 1}u] = hat;
  fits[fitsOffset + index * RECORD_WORDS + ${size + 2}u] = f32(code);`;
}
