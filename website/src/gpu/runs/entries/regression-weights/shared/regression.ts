// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared pieces of the regression runs: Chicago tracts with two observed demographic predictors, a response that
// carries chosen spatial dependence, and the two panels every run draws (a tract map of a per-row
// output on the left, coefficient bars with the true values on the right).

import type {GPUSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import {createTracts, type CsrWeights, type TractsInput} from '../../../inputs';
import {addTableColumn, uploadWeightsBuffers, type WeightsBuffers} from './weights';

/** The coefficients the response is generated with: intercept, then one per predictor. */
export const TRUE_COEFFICIENTS = [1, 2, -1.5] as const;

/** Share of the view width the map panel uses; the chart panel takes the rest. */
export const MAP_SHARE = 0.58;

export type RegressionInputs = {
  tracts: TractsInput;
  /** Width of the map panel in world units. */
  mapWidth: number;
  /** Row-standardized rook contiguity. */
  weights: CsrWeights;
  /** Row-major `[x1, x2]` per tract. */
  predictors: Float32Array;
  /** Fixed standard normal draws, scaled by the noise slider. */
  normals: Float32Array;
  /** Seeded uniform draws per tract, for sampling counts and outcomes. */
  uniforms: Float32Array;
  /** Region (0 or 1) per tract, for the regime models: west and east. */
  regimes: Uint32Array;
};

/** Chicago tract predictors scaled to the panel range so the coefficient experiments share a scale. */
export function createRegressionInputs(
  random: () => number,
  world: {width: number; height: number},
  rows = 10,
  share = MAP_SHARE
): RegressionInputs {
  const tracts = createTracts(random, {width: world.width * share, height: world.height}, rows);
  const count = tracts.count;
  const predictors = new Float32Array(count * 2);
  const olderShare = Float32Array.from(
    tracts.age65,
    (older, tract) => older / tracts.population[tract]
  );
  const minimum = Math.min(...olderShare);
  const range = Math.max(...olderShare) - minimum;
  for (let tract = 0; tract < count; tract++) {
    predictors[tract * 2] = tracts.values[tract];
    predictors[tract * 2 + 1] = range > 0 ? (olderShare[tract] - minimum) / range : 0;
  }
  const normals = Float32Array.from({length: count}, () => {
    let sum = 0;
    for (let draw = 0; draw < 6; draw++) sum += random();
    return (sum - 3) / Math.sqrt(0.5);
  });
  const uniforms = Float32Array.from({length: count}, () => random());
  const weights = standardizeRows(tracts.contiguity);
  const regimes = Uint32Array.from({length: count}, (_, tract) =>
    tracts.centers[tract * 2] < (world.width * share) / 2 ? 0 : 1
  );
  return {
    tracts,
    mapWidth: world.width * share,
    weights,
    predictors,
    normals,
    uniforms,
    regimes
  };
}

/** Copy of the CSR with every row's weights summing to one. */
export function standardizeRows(csr: CsrWeights): CsrWeights {
  const weights = new Float32Array(csr.weights.length);
  for (let row = 0; row < csr.rowCount; row++) {
    const count = csr.offsets[row + 1] - csr.offsets[row];
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      weights[slot] = 1 / count;
    }
  }
  return {...csr, weights};
}

/** `W x` with the CSR weights. */
function multiplyWeights(csr: CsrWeights, vector: Float32Array): Float32Array {
  const result = new Float32Array(csr.rowCount);
  for (let row = 0; row < csr.rowCount; row++) {
    let sum = 0;
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      sum += csr.weights[slot] * vector[csr.neighbors[slot]];
    }
    result[row] = sum;
  }
  return result;
}

/** Solves `(I - rho W) z = right` by the Neumann series, fine for a row-standardized W and rho < 1. */
function solveSpatialSeries(csr: CsrWeights, right: Float32Array, rho: number): Float32Array {
  let term = Float32Array.from(right);
  const total = Float32Array.from(right);
  for (let power = 1; power < 40; power++) {
    term = multiplyWeights(csr, term).map(value => value * rho);
    for (let row = 0; row < total.length; row++) total[row] += term[row];
  }
  return total;
}

export type ResponseOptions = {
  noise: number;
  /** Spatial lag: y = rho W y + X b + e. */
  lagRho?: number;
  /** Spatial error: y = X b + (I - lambda W)^-1 e. */
  errorLambda?: number;
  /** Replaces the true coefficients, for example by region. */
  coefficients?: (tract: number) => readonly [number, number, number];
};

/** The response of the data generating process, from the true coefficients and the slider values. */
export function createResponse(inputs: RegressionInputs, options: ResponseOptions): Float32Array {
  const count = inputs.tracts.count;
  const signal = new Float32Array(count);
  for (let tract = 0; tract < count; tract++) {
    const [intercept, first, second] = options.coefficients?.(tract) ?? TRUE_COEFFICIENTS;
    signal[tract] =
      intercept + first * inputs.predictors[tract * 2] + second * inputs.predictors[tract * 2 + 1];
  }
  const errors = inputs.normals.map(value => value * options.noise);
  if (options.errorLambda) {
    const correlated = solveSpatialSeries(inputs.weights, errors, options.errorLambda);
    return signal.map((value, tract) => value + correlated[tract]);
  }
  return solveSpatialSeries(
    inputs.weights,
    signal.map((value, tract) => value + errors[tract]),
    options.lagRho ?? 0
  );
}

/** Uploads the row-standardized weights as the weights input of a regression operation. */
export function uploadStandardizedWeights(
  builder: RunBuilder,
  csr: CsrWeights
): Omit<WeightsBuffers, 'overflow' | 'port'> & {input: GPUSpatialWeights} {
  const uploaded = uploadWeightsBuffers(builder, 'weights', csr);
  uploaded.input.distances = undefined;
  uploaded.input.information = {
    rowCount: csr.rowCount,
    transform: 'row',
    symmetric: false,
    provenance: {source: 'upload'}
  };
  return uploaded;
}

/** The tract mesh buffers and the layers that draw per-tract values on them. */
export function uploadRegressionMap(builder: RunBuilder, inputs: RegressionInputs, offsetX = 0) {
  const {tracts} = inputs;
  const shift = (values: Float32Array, stride: number) =>
    offsetX === 0
      ? values
      : Float32Array.from(values, (value, index) =>
          index % stride === 0 ? value + offsetX : value
        );
  const suffix = offsetX === 0 ? '' : '-shifted';
  return {
    vertices: builder.upload(`tract-vertices${suffix}`, 'float32x2', shift(tracts.vertices, 2)),
    indices: builder.upload(`tract-indices${suffix}`, 'uint32', tracts.triangleIndices),
    owners: builder.upload(`tract-owners${suffix}`, 'uint32', tracts.triangleOwners),
    centers: builder.upload(`tract-centers${suffix}`, 'float32x2', shift(tracts.centers, 2)),
    outline: builder.upload(`tract-outline${suffix}`, 'float32x4', shift(tracts.outline, 2))
  };
}

/** A tract map layer of one per-tract buffer, plus the tract outline. */
export function drawTractMap(
  inputs: RegressionInputs,
  mesh: ReturnType<typeof uploadRegressionMap>,
  values: RunBuffer,
  look: {
    ramp?: 'sequential' | 'diverging' | 'categorical';
    range?: readonly [number, number] | 'auto';
    logarithmic?: boolean;
  } = {}
): RunLayer[] {
  return [
    {
      layer: 'polygons',
      positions: mesh.vertices,
      indices: mesh.indices,
      owners: mesh.owners,
      values,
      triangleCount: inputs.tracts.triangleCount,
      ramp: look.ramp ?? 'diverging',
      range: look.range ?? 'auto',
      logarithmic: look.logarithmic
    },
    {layer: 'lines', context: true, segments: mesh.outline, count: inputs.tracts.outlineCount}
  ];
}

/** Buffers behind the coefficient panel. */
export type CoefficientPanel = {
  estimates: RunBuffer<'float32'>;
  /** Standard errors, when the table has a second column. */
  standardErrors?: RunBuffer<'float32'>;
  truth: RunBuffer<'float32'>;
  positions: RunBuffer<'float32'>;
  /** Zeros for the zero line of the chart. */
  zeros: RunBuffer<'float32'>;
};

/**
 * Uploads the true coefficients and marker positions of the chart panel, and extracts the estimate
 * column of a table the operation wrote (`columnCount` floats per coefficient row).
 */
export function createCoefficientPanel(
  builder: RunBuilder,
  table: RunBuffer<'float32'>,
  columnCount: number,
  truth: readonly number[]
): CoefficientPanel {
  return {
    estimates: addTableColumn(builder, 'coefficients', table, columnCount, 0, truth.length),
    standardErrors:
      columnCount > 1
        ? addTableColumn(builder, 'standard-errors', table, columnCount, 1, truth.length)
        : undefined,
    truth: builder.upload('true-coefficients', 'float32', Float32Array.from(truth)),
    positions: builder.upload(
      'coefficient-positions',
      'float32',
      Float32Array.from(truth, (_, index) => (index + 0.5) / truth.length)
    ),
    zeros: builder.upload('coefficient-zeros', 'float32', new Float32Array(64))
  };
}

/** The shared range of the coefficient chart. */
export const COEFFICIENT_RANGE = [-3, 3] as const;

const POSITIVE_BAR_COLOR = [0.4, 0.68, 0.98, 0.95] as const;
const NEGATIVE_BAR_COLOR = [0.98, 0.55, 0.32, 0.95] as const;

type ChartRectangle = readonly [number, number, number, number];

/**
 * Signed bars that grow from the zero line: positive estimates up (blue), negative ones down
 * (orange). The bars layer always grows from the bottom of its rectangle, so each half is its own
 * layer, the lower one on a flipped rectangle.
 */
export function drawSignedBars(
  values: RunBuffer,
  rectangle: ChartRectangle,
  limit: number,
  color?: readonly [number, number, number, number]
): RunLayer[] {
  const [left, bottom, right, top] = rectangle;
  const zero = (bottom + top) / 2;
  return [
    {
      layer: 'chart',
      kind: 'bars',
      values,
      color: color ?? POSITIVE_BAR_COLOR,
      range: [0, limit],
      rectangle: [left, zero, right, top]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values,
      color: color ?? NEGATIVE_BAR_COLOR,
      range: [0, -limit],
      rectangle: [left, zero, right, bottom]
    }
  ];
}

/** A horizontal reference line at zero across a chart rectangle whose y range is symmetric. */
export function drawZeroLine(zeros: RunBuffer, rectangle: ChartRectangle): RunLayer {
  return {
    layer: 'chart',
    kind: 'line',
    values: zeros,
    range: COEFFICIENT_RANGE,
    color: [1, 1, 1, 0.4],
    radius: 0.9,
    rectangle
  };
}

/** Coefficient bars (estimates) with the true values as dots, and standard errors below them. */
export function drawCoefficientPanel(
  panel: CoefficientPanel,
  options: {rectangle?: ChartRectangle} = {}
): RunLayer[] {
  const left = MAP_SHARE + 0.08;
  const rectangle =
    options.rectangle ??
    (panel.standardErrors
      ? ([left, 0.42, 0.97, 0.9] as const)
      : ([left, 0.14, 0.97, 0.86] as const));
  const layers: RunLayer[] = [
    ...drawSignedBars(panel.estimates, rectangle, COEFFICIENT_RANGE[1]),
    drawZeroLine(panel.zeros, rectangle),
    {
      layer: 'chart',
      kind: 'scatter',
      values: panel.truth,
      xValues: panel.positions,
      xRange: [0, 1],
      range: COEFFICIENT_RANGE,
      color: [1, 1, 1, 0.95],
      radius: 5,
      rectangle
    }
  ];
  if (panel.standardErrors) {
    layers.push({
      layer: 'chart',
      kind: 'bars',
      values: panel.standardErrors,
      ramp: 'sequential',
      range: [0, 0.6],
      color: [0.55, 0.65, 0.95, 0.9],
      rectangle: [left, 0.08, 0.97, 0.34]
    });
  }
  return layers;
}

/** The symmetric range of the residual maps, so noise and strength change what the map shows. */
export const RESIDUAL_RANGE = [-1.5, 1.5] as const;

/** True coefficients that drift from west to east: the slope of the first predictor varies. */
export function getDriftingCoefficients(
  inputs: RegressionInputs,
  strength: number
): (tract: number) => readonly [number, number, number] {
  return tract => [
    TRUE_COEFFICIENTS[0],
    0.5 + strength * (inputs.tracts.centers[tract * 2] / inputs.mapWidth),
    TRUE_COEFFICIENTS[2]
  ];
}

/** Two maps side by side (each half the width) for the local regression runs. */
export const DUAL_MAP_SHARE = 0.44;

/** A tract map of residuals on the fixed symmetric range. */
export function drawResidualMap(
  inputs: RegressionInputs,
  mesh: ReturnType<typeof uploadRegressionMap>,
  residuals: RunBuffer
): RunLayer[] {
  return drawTractMap(inputs, mesh, residuals, {ramp: 'diverging', range: RESIDUAL_RANGE});
}
