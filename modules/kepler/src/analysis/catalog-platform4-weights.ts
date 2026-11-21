// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError} from './catalog-gpu-session';
import {
  runChunkedWeightsAlgebraGpu,
  runChunkedWeightsTransformGpu,
  runChunkedWeightsTransposeGpu
} from './catalog-platform5-weights-chunked';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  type Values
} from './catalog-platform3-b-common';
import {makeLocalFrame} from './catalog-local-frame';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

const MAXIMUM_WEIGHT_POINTS = 4000;
const MAXIMUM_EDGE_ROWS = 200000;

/** Compressed sparse row weights; neighbors are ascending within a row and never list the row. */
export type WeightsCsr = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

type WeightPoints = {
  x: Float64Array;
  y: Float64Array;
  featureIndex: Int32Array;
  coordinates: [number, number][];
  count: number;
  droppedCount: number;
};

function readWeightPoints(
  collection: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): WeightPoints {
  const kept: {feature: number; position: [number, number]}[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const position = getPointPositions(feature.geometry)[0];
    if (position && Number.isFinite(position[0]) && Number.isFinite(position[1]))
      kept.push({feature: index, position: [position[0], position[1]]});
    else droppedCount++;
  });
  // The pairwise neighbor search is quadratic, so the input is capped.
  if (kept.length > MAXIMUM_WEIGHT_POINTS)
    throw new RangeError(`Weights accept at most ${MAXIMUM_WEIGHT_POINTS} points`);
  const frame = makeLocalFrame(
    coordinateSystem,
    kept.map(entry => entry.position),
    {shift: 'minimum'}
  );
  const x = new Float64Array(kept.length);
  const y = new Float64Array(kept.length);
  kept.forEach((entry, index) => {
    [x[index], y[index]] = frame.forward(entry.position);
  });
  return {
    x,
    y,
    featureIndex: Int32Array.from(kept.map(entry => entry.feature)),
    coordinates: kept.map(entry => entry.position),
    count: kept.length,
    droppedCount
  };
}

function makeCsr(rows: {id: number; weight: number}[][]): WeightsCsr {
  const offsets = new Uint32Array(rows.length + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  rows.forEach((row, index) => {
    for (const entry of row.sort((a, b) => a.id - b.id)) {
      neighbors.push(entry.id);
      weights.push(Math.fround(entry.weight));
    }
    offsets[index + 1] = neighbors.length;
  });
  return {
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

/** k nearest neighbors per point (ties by index); weights are 1 or 1 / (1 + d / decay). */
export function buildNearestNeighborWeights(
  points: WeightPoints,
  neighborCount: number,
  decayDistance: number
): WeightsCsr {
  const rows: {id: number; weight: number}[][] = [];
  for (let row = 0; row < points.count; row++) {
    const candidates: {id: number; distance: number}[] = [];
    for (let other = 0; other < points.count; other++)
      if (other !== row)
        candidates.push({
          id: other,
          distance: Math.hypot(points.x[other] - points.x[row], points.y[other] - points.y[row])
        });
    candidates.sort((a, b) => a.distance - b.distance || a.id - b.id);
    rows.push(
      candidates.slice(0, neighborCount).map(candidate => ({
        id: candidate.id,
        weight: decayDistance > 0 ? 1 / (1 + candidate.distance / decayDistance) : 1
      }))
    );
  }
  return makeCsr(rows);
}

/** Neighbors within a distance band; weights 1 / (1 + d / radius). */
export function buildDistanceBandWeights(points: WeightPoints, radius: number): WeightsCsr {
  const rows: {id: number; weight: number}[][] = [];
  for (let row = 0; row < points.count; row++) {
    const entries: {id: number; weight: number}[] = [];
    for (let other = 0; other < points.count; other++) {
      if (other === row) continue;
      const distance = Math.hypot(points.x[other] - points.x[row], points.y[other] - points.y[row]);
      if (distance <= radius) entries.push({id: other, weight: 1 / (1 + distance / radius)});
    }
    rows.push(entries);
  }
  return makeCsr(rows);
}

// ---------------------------------------------------------------------------------------------
// CPU references

export type WeightsCombineRule = 'left' | 'right' | 'sum' | 'min' | 'max' | 'product' | 'binary';
export type WeightsSetOperation = 'union' | 'intersection' | 'difference' | 'symmetric-difference';

export function computeWeightsAlgebra(
  left: WeightsCsr,
  right: WeightsCsr,
  operation: WeightsSetOperation,
  rule: WeightsCombineRule
): WeightsCsr {
  validateRequirement(
    'weights-algebra',
    ['union', 'intersection', 'difference', 'symmetric-difference'].includes(operation),
    `operation must be union, intersection, difference or symmetric-difference; received ${String(operation)}`
  );
  validateRequirement(
    'weights-algebra',
    ['left', 'right', 'sum', 'min', 'max', 'product', 'binary'].includes(rule),
    `weightRule must be left, right, sum, min, max, product or binary; received ${rule}`
  );
  const rowCount = left.offsets.length - 1;
  const rows: {id: number; weight: number}[][] = [];
  const getRow = (csr: WeightsCsr, row: number) =>
    new Map(
      Array.from(csr.neighbors.subarray(csr.offsets[row], csr.offsets[row + 1]), (id, slot) => [
        id,
        csr.weights[csr.offsets[row] + slot]
      ])
    );
  for (let row = 0; row < rowCount; row++) {
    const leftRow = getRow(left, row);
    const rightRow = getRow(right, row);
    const ids = [...new Set([...leftRow.keys(), ...rightRow.keys()])].sort((a, b) => a - b);
    const entries: {id: number; weight: number}[] = [];
    for (const id of ids) {
      const a = leftRow.get(id);
      const b = rightRow.get(id);
      const keep = {
        union: true,
        intersection: a !== undefined && b !== undefined,
        difference: a !== undefined && b === undefined,
        'symmetric-difference': (a !== undefined) !== (b !== undefined)
      }[operation];
      if (!keep) continue;
      let weight: number;
      if (rule === 'binary') weight = 1;
      else if (a !== undefined && b !== undefined)
        weight = {
          left: a,
          right: b,
          sum: a + b,
          min: Math.min(a, b),
          max: Math.max(a, b),
          product: a * b
        }[rule];
      else weight = (a ?? b) as number;
      entries.push({id, weight});
    }
    rows.push(entries);
  }
  return makeCsr(rows);
}

export function computeWeightsTranspose(csr: WeightsCsr): WeightsCsr {
  const rowCount = csr.offsets.length - 1;
  const rows: {id: number; weight: number}[][] = Array.from({length: rowCount}, () => []);
  for (let row = 0; row < rowCount; row++)
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++)
      rows[csr.neighbors[slot]].push({id: row, weight: csr.weights[slot]});
  return makeCsr(rows);
}

export type WeightsTransformOperation = 'row' | 'binary' | 'double' | 'variance';

/** PySAL transforms R, B, D and V in float64; returns one weight per slot. */
export function computeWeightsTransform(
  csr: WeightsCsr,
  operation: WeightsTransformOperation
): Float64Array {
  const rowCount = csr.offsets.length - 1;
  const result = Float64Array.from(csr.weights);
  if (operation === 'double') {
    const total = csr.weights.reduce((sum, weight) => sum + weight, 0);
    return result.map(weight => (total > 0 ? weight / total : 0));
  }
  if (operation === 'variance') {
    const norms = new Float64Array(rowCount);
    let scaledTotal = 0;
    for (let row = 0; row < rowCount; row++) {
      let squares = 0;
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++)
        squares += csr.weights[slot] ** 2;
      norms[row] = Math.sqrt(squares);
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++)
        scaledTotal += norms[row] > 0 ? csr.weights[slot] / norms[row] : 0;
    }
    for (let row = 0; row < rowCount; row++)
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++)
        result[slot] =
          norms[row] > 0 && scaledTotal > 0
            ? (csr.weights[slot] / norms[row]) * (rowCount / scaledTotal)
            : 0;
    return result;
  }
  for (let row = 0; row < rowCount; row++) {
    let sum = 0;
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) sum += csr.weights[slot];
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++)
      result[slot] =
        operation === 'binary'
          ? csr.weights[slot] > 0
            ? 1
            : 0
          : sum > 0
            ? csr.weights[slot] / sum
            : 0;
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Output

function buildEdgeOutput(
  points: WeightPoints,
  csr: WeightsCsr,
  weights: ArrayLike<number>,
  summary: CatalogRunOutput['summary'],
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (points.droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position are skipped.',
      points.droppedCount
    );
  const table = makeCatalogTable(
    'weights-edges',
    'Spatial weights edges',
    {
      source_index: 'int32',
      target_index: 'int32',
      source_feature_index: 'int32',
      target_feature_index: 'int32',
      weight: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let islandCount = 0;
  for (let row = 0; row < points.count; row++) {
    if (csr.offsets[row] === csr.offsets[row + 1]) islandCount++;
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      const target = csr.neighbors[slot];
      appendCatalogRow(table, {
        source_index: row,
        target_index: target,
        source_feature_index: points.featureIndex[row],
        target_feature_index: points.featureIndex[target],
        weight: weights[slot],
        geometry: makeCatalogGeometryText('LineString', [
          points.coordinates[row],
          points.coordinates[target]
        ])
      });
    }
  }
  return {
    tables: [table],
    diagnostics,
    summary: {...summary, edgeCount: table.rowCount, rowCount: points.count, islandCount},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const POINTS_INPUT = makeInputSpec(
  'points',
  'Points',
  ['point'],
  MAXIMUM_WEIGHT_POINTS,
  MAXIMUM_WEIGHT_POINTS
);

const WEIGHTS_BEHAVIOR = [
  'Weights are built from the first position of each point feature in a local meter frame (planar units for planar data); features without a position are skipped and counted; more than 4,000 points throws (the neighbor search is pairwise).',
  'Rows list neighbors ascending, never themselves; float32 weights; a row without neighbors is an island and is counted.',
  'More than 200,000 edge rows throws.'
];

const CHUNK_ROWS_PARAMETER: CatalogParameterSpec = makeNumberParameter(
  'chunkRows',
  'Rows per GPU chunk (0 = automatic)',
  0,
  0,
  1e6,
  undefined,
  'integer'
);

const NEIGHBOR_PARAMETERS: CatalogParameterSpec[] = [
  makeNumberParameter('neighborCount', 'Nearest neighbors k', 4, 1, 50, undefined, 'integer'),
  makeNumberParameter('decayDistance', 'Distance decay (0 = binary weights)', 0, 0, 1e9, 'crs'),
  COORDINATE_SYSTEM_PARAMETER,
  CHUNK_ROWS_PARAMETER
];

const EDGE_OUTPUT = makeOutputSpec(
  'weights-edges',
  'Spatial weights edges',
  'line',
  ['source_index', 'target_index'],
  MAXIMUM_EDGE_ROWS
);

function readOperands(inputs: Readonly<Record<string, CatalogFeatureCollection>>, values: Values) {
  const points = readWeightPoints(
    inputs['points'],
    values['coordinateSystem'] as CatalogCoordinateSystem
  );
  return points;
}

// ---------------------------------------------------------------------------------------------
// weights-algebra

const ALGEBRA_PARAMETERS: CatalogParameterSpec[] = [
  makeNumberParameter(
    'neighborCount',
    'Left: nearest neighbors k (weight 1)',
    4,
    1,
    50,
    undefined,
    'integer'
  ),
  makeNumberParameter(
    'distanceBand',
    'Right: distance band (weight 1 / (1 + d / band))',
    500,
    1e-6,
    1e9,
    'crs'
  ),
  {
    name: 'operation',
    label: 'Set operation',
    kind: 'enum',
    options: ['union', 'intersection', 'difference', 'symmetric-difference'],
    defaultValue: 'union'
  },
  {
    name: 'weightRule',
    label: 'Weight rule for shared neighbors',
    kind: 'enum',
    options: ['binary', 'left', 'right', 'sum', 'min', 'max', 'product'],
    defaultValue: 'binary'
  },
  COORDINATE_SYSTEM_PARAMETER,
  CHUNK_ROWS_PARAMETER
];

function buildAlgebraOperands(points: WeightPoints, values: Values) {
  return {
    left: buildNearestNeighborWeights(points, values['neighborCount'] as number, 0),
    right: buildDistanceBandWeights(points, values['distanceBand'] as number)
  };
}

const WEIGHTS_ALGEBRA_BINDING: CatalogOperationBinding = {
  id: 'weights-algebra',
  title: 'Spatial weights set algebra',
  domain: 'analysis',
  summary:
    'Combines a k-nearest-neighbor weights matrix and a distance-band weights matrix with union, intersection, difference or symmetric difference.',
  capabilityIds: ['spatial-neighborhoods'],
  contributorNames: ['GPUSpatialWeightsAlgebra'],
  gpuContributorNames: ['GPUChunkedSpatialWeightsAlgebra'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: ALGEBRA_PARAMETERS,
  outputs: [EDGE_OUTPUT],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHTS_BEHAVIOR,
    'symmetric-difference keeps each neighbor present in exactly one operand.',
    'Neighbors present in both operands take the chosen rule (binary gives 1); neighbors present in one operand keep that operand weight (1 for the left, the decayed weight for the right), except for rule binary.'
  ],
  limitation:
    'The GPU engine runs GPUChunkedSpatialWeightsAlgebra over row chunks (chunkRows, automatic by default: one chunk unless a chunk would exceed one storage binding); the higher-order, block, subgraph and self-weight operations are not exposed. Both operands are built on the CPU from the same points; chunk outputs are joined on the CPU for the table.',
  run: (inputs, values) => {
    const points = readOperands(inputs, values);
    const {left, right} = buildAlgebraOperands(points, values);
    const result = computeWeightsAlgebra(
      left,
      right,
      values['operation'] as WeightsSetOperation,
      values['weightRule'] as WeightsCombineRule
    );
    return buildEdgeOutput(points, result, result.weights, {
      operation: values['operation'] as string
    });
  },
  runGpu: async (context, inputs, values) => {
    const points = readOperands(inputs, values);
    if (!points.count) throw new CatalogGpuUnsupportedError('No points');
    const {left, right} = buildAlgebraOperands(points, values);
    const {csr, chunkCount} = await runChunkedWeightsAlgebraGpu(
      context.device,
      left,
      right,
      values['operation'] as WeightsSetOperation,
      values['weightRule'] as WeightsCombineRule,
      values['chunkRows'] as number
    );
    return buildEdgeOutput(
      points,
      csr,
      csr.weights,
      {operation: values['operation'] as string, chunkCount},
      ['GPUChunkedSpatialWeightsAlgebra']
    );
  }
};

// ---------------------------------------------------------------------------------------------
// weights-transpose

const WEIGHTS_TRANSPOSE_BINDING: CatalogOperationBinding = {
  id: 'weights-transpose',
  title: 'Spatial weights transpose',
  domain: 'analysis',
  summary:
    'Builds directed k-nearest-neighbor weights and publishes the transposed matrix (who lists me as a neighbor).',
  capabilityIds: ['spatial-neighborhoods'],
  contributorNames: ['GPUSpatialWeightsTranspose'],
  gpuContributorNames: ['GPUChunkedSpatialWeightsTranspose'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: NEIGHBOR_PARAMETERS,
  outputs: [EDGE_OUTPUT],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHTS_BEHAVIOR,
    'Edge (i, j) of the input appears as (j, i) in the output with the same weight; k-nearest-neighbor weights are generally asymmetric.'
  ],
  limitation:
    'Square matrices only; the input is chunked (GPUChunkedSpatialWeightsTranspose, chunkRows) but the transpose output is one whole CSR that must fit one storage binding.',
  run: (inputs, values) => {
    const points = readOperands(inputs, values);
    const csr = buildNearestNeighborWeights(
      points,
      values['neighborCount'] as number,
      values['decayDistance'] as number
    );
    const result = computeWeightsTranspose(csr);
    return buildEdgeOutput(points, result, result.weights, {});
  },
  runGpu: async (context, inputs, values) => {
    const points = readOperands(inputs, values);
    if (!points.count) throw new CatalogGpuUnsupportedError('No points');
    const csr = buildNearestNeighborWeights(
      points,
      values['neighborCount'] as number,
      values['decayDistance'] as number
    );
    const {csr: result, chunkCount} = await runChunkedWeightsTransposeGpu(
      context.device,
      csr,
      values['chunkRows'] as number
    );
    return buildEdgeOutput(points, result, result.weights, {chunkCount}, [
      'GPUChunkedSpatialWeightsTranspose'
    ]);
  }
};

// ---------------------------------------------------------------------------------------------
// weights-transform

const WEIGHTS_TRANSFORM_BINDING: CatalogOperationBinding = {
  id: 'weights-transform',
  title: 'Spatial weights standardization',
  domain: 'analysis',
  summary:
    'Builds k-nearest-neighbor weights and applies row, binary, double or variance standardization (PySAL R, B, D, V).',
  capabilityIds: ['spatial-neighborhoods'],
  contributorNames: ['GPUSpatialWeightsTransform'],
  gpuContributorNames: ['GPUChunkedSpatialWeightsTransform'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: [
    ...NEIGHBOR_PARAMETERS,
    {
      name: 'operation',
      label: 'Transform',
      kind: 'enum',
      options: ['row', 'binary', 'double', 'variance'],
      defaultValue: 'row'
    }
  ],
  outputs: [EDGE_OUTPUT],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHTS_BEHAVIOR,
    'Row standardization leaves an all-zero row at zero; double divides by the global sum, variance follows libpysal V.'
  ],
  limitation:
    'Runs GPUChunkedSpatialWeightsTransform over row chunks (chunkRows); double and variance sums are float32 partials combined in a fixed order. The kernel and average-reciprocal transforms are not exposed (kernel needs distances, and average-reciprocal uses a whole scratch CSR).',
  run: (inputs, values) => {
    const points = readOperands(inputs, values);
    const csr = buildNearestNeighborWeights(
      points,
      values['neighborCount'] as number,
      values['decayDistance'] as number
    );
    return buildEdgeOutput(
      points,
      csr,
      computeWeightsTransform(csr, values['operation'] as WeightsTransformOperation),
      {operation: values['operation'] as string}
    );
  },
  runGpu: async (context, inputs, values) => {
    const points = readOperands(inputs, values);
    if (!points.count) throw new CatalogGpuUnsupportedError('No points');
    const csr = buildNearestNeighborWeights(
      points,
      values['neighborCount'] as number,
      values['decayDistance'] as number
    );
    if (!csr.neighbors.length) throw new CatalogGpuUnsupportedError('No weight slots');
    const {weights, chunkCount} = await runChunkedWeightsTransformGpu(
      context.device,
      csr,
      values['operation'] as WeightsTransformOperation,
      values['chunkRows'] as number
    );
    return buildEdgeOutput(
      points,
      csr,
      weights,
      {operation: values['operation'] as string, chunkCount},
      ['GPUChunkedSpatialWeightsTransform']
    );
  }
};

export const PLATFORM4_WEIGHTS_BINDINGS: readonly CatalogOperationBinding[] = [
  WEIGHTS_ALGEBRA_BINDING,
  WEIGHTS_TRANSPOSE_BINDING,
  WEIGHTS_TRANSFORM_BINDING
];
