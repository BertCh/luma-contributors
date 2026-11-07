// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/graph';
import {runExample as runKriging} from './local-kriging.example';
import {runExample as runRadialBasis} from './local-rbf.example';
import {runExample as runNaturalNeighbor} from './natural-neighbor.example';
import {runExample as runSimulation} from './conditional-gaussian-simulation.example';
import {runS2Lens, type S2CaptureSpec, type S2Uploads} from './s2-lens-run';

export type InterpolationLensKind =
  | 'local-kriging'
  | 'local-rbf'
  | 'natural-neighbor'
  | 'conditional-gaussian-simulation';
const ELEMENTS = {
  'local-kriging': 'GPULocalKriging',
  'local-rbf': 'GPULocalRBF',
  'natural-neighbor': 'GPUNaturalNeighbor',
  'conditional-gaussian-simulation': 'GPUConditionalGaussianSimulation'
} as const;
const QUERY_COUNT = 4;
const REPLICATE_COUNT = 16;

/** Construct real GPU predictions from explicitly synthetic planar sites and caller-owned uploads. */
export async function buildInterpolationLens(
  kind: InterpolationLensKind,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const coordinates = [0, 0, 2, 0, 2, 2, 0, 2, 0.6, 0.7, 1.4, 1.2];
  const sampleValues = Array.from(
    {length: 6},
    (_, row) => 2 + 0.5 * coordinates[row * 2] - 0.25 * coordinates[row * 2 + 1]
  );
  const positions = uploads.view('positions', Float32Array.from(coordinates), 'float32x2');
  const values = uploads.view('sample-values', Float32Array.from(sampleValues), 'float32');
  const queries = uploads.view(
    'queries',
    new Float32Array([0.8, 1.1, 1.1, 0.4, 0.6, 0.7, 1.7, 1.6]),
    'float32x2'
  );
  const captures: Record<string, S2CaptureSpec> = {
    positions: {
      view: positions,
      role: 'context',
      keySpace: 'sample',
      geometry: {
        kind: 'points',
        coordinateModel: 'projected',
        unit: 'arbitrary planar units',
        valuesCapture: 'sampleValues'
      }
    },
    sampleValues: {
      view: values,
      role: 'input',
      keySpace: 'sample',
      meaning: 'Synthetic affine response: 2 + 0.5x - 0.25y'
    },
    queries: {
      view: queries,
      role: 'output',
      keySpace: 'query',
      geometry: {
        kind: 'points',
        coordinateModel: 'projected',
        unit: 'arbitrary planar units',
        valuesCapture: 'predictions'
      }
    }
  };
  const output = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number,
    keySpace: string,
    meaning: string
  ) => {
    const view = createTransientView(graph, name, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
    captures[name] = {view, role: 'output', keySpace, meaning};
    return view;
  };
  const covariance = {model: 'exponential' as const, range: 3, sill: 2, nugget: 0.1};
  const observations = () => {
    const view = uploads.view(
      'observations',
      Float32Array.from(sampleValues.flatMap(value => [value, 0.1, 0, 1])),
      'float32x4'
    );
    captures['observations'] = {
      view,
      role: 'input',
      keySpace: 'sample',
      meaning: '[response, measurement-error variance, variable index, validity]'
    };
    return view;
  };
  let predictionSource: GraphDataView<'float32' | 'float32x4'>;
  if (kind === 'conditional-gaussian-simulation') {
    const realizations = output(
      'values',
      'float32',
      QUERY_COUNT * REPLICATE_COUNT,
      'realization',
      'Replicate-major latent query realizations'
    );
    const moments = output(
      'moments',
      'float32x2',
      QUERY_COUNT,
      'query',
      '[conditional mean, conditional variance]'
    );
    const status = output(
      'statuses',
      'uint32',
      REPLICATE_COUNT,
      'replicate',
      '0 success; 1 invalid input; 2 factorization failure'
    );
    await runSimulation(graph, {
      positions,
      observations: observations(),
      queries,
      covariance,
      mean: 2,
      seed: 347,
      replicateCount: REPLICATE_COUNT,
      output: {values: realizations, moments, status}
    });
    predictionSource = realizations;
  } else {
    const result = output(
      'results',
      'float32x4',
      QUERY_COUNT,
      'query',
      kind === 'local-kriging'
        ? '[prediction, latent variance, pivot ratio, retained count]'
        : kind === 'local-rbf'
          ? '[prediction, pivot ratio, retained count, reserved]'
          : '[prediction, query cell area, weight sum, contributing count]'
    );
    const status = output(
      'statuses',
      'uint32',
      QUERY_COUNT,
      'query',
      '0 success; 1 invalid; 2 degenerate; natural neighbor also uses 3 for capacity'
    );
    if (kind === 'local-kriging') {
      await runKriging(graph, {
        positions,
        observations: observations(),
        supports: queries,
        covariance,
        neighborCount: 6,
        mode: 'universal',
        output: {values: result, status}
      });
    } else if (kind === 'local-rbf') {
      await runRadialBasis(graph, {
        positions,
        values,
        queries,
        kernel: 'thin-plate-spline',
        smoothing: 0.02,
        neighborCount: 6,
        output: {values: result, status}
      });
    } else {
      const weights = output(
        'weights',
        'float32',
        QUERY_COUNT * 6,
        'query-sample',
        'Query-major Sibson weights in original sample order'
      );
      await runNaturalNeighbor(graph, {
        positions,
        values,
        queries,
        output: {values: result, status, weights}
      });
    }
    predictionSource = result;
  }
  const predictions = output(
    'predictions',
    'float32',
    QUERY_COUNT,
    'query',
    kind === 'conditional-gaussian-simulation'
      ? 'First joint realization (replicate 0)'
      : 'Predicted scalar response'
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'capture-predictions',
      operation: 'InterpolationLensPredictionCapture',
      invocationCount: QUERY_COUNT,
      bindings: [
        {name: 'source', view: predictionSource, type: 'f32', access: 'read'},
        {name: 'destination', view: predictions, type: 'f32', access: 'read_write'}
      ],
      body: `destination[destinationOffset + index] = source[sourceOffset + index * ${kind === 'conditional-gaussian-simulation' ? 1 : 4}u];`
    })
  );
  return captures;
}

/** Execute, capture and release the same graph used by the public API examples. */
export function runInterpolationLens(device: Device, kind: InterpolationLensKind) {
  return runS2Lens(device, {
    scenarioId: `gpu-spatial-analysis.${kind}`,
    element: ELEMENTS[kind],
    datasets: ['synthetic-interpolation-six-sites-v1'],
    seed: kind === 'conditional-gaussian-simulation' ? 347 : undefined,
    retain:
      kind === 'conditional-gaussian-simulation'
        ? ['values', 'moments', 'statuses', 'predictions']
        : kind === 'natural-neighbor'
          ? ['results', 'statuses', 'weights', 'predictions']
          : ['results', 'statuses', 'predictions'],
    keySpaces: [
      {id: 'query', label: 'Prediction site', size: QUERY_COUNT, indexSpace: '1d'},
      {id: 'sample', label: 'Synthetic observation', size: 6, indexSpace: '1d'},
      {
        id: 'query-sample',
        label: 'Query × sample weight',
        size: QUERY_COUNT * 6,
        shape: [6, QUERY_COUNT],
        indexSpace: '2d'
      },
      {id: 'replicate', label: 'Joint realization', size: REPLICATE_COUNT, indexSpace: '1d'},
      {
        id: 'realization',
        label: 'Replicate × query',
        size: QUERY_COUNT * REPLICATE_COUNT,
        shape: [QUERY_COUNT, REPLICATE_COUNT],
        indexSpace: '2d'
      }
    ],
    choices:
      kind === 'local-kriging'
        ? {mode: 'universal', neighborCount: 6, supportCount: 1}
        : kind === 'local-rbf'
          ? {kernel: 'thin-plate-spline', smoothing: 0.02}
          : kind === 'natural-neighbor'
            ? {maximumVertexCount: 64}
            : {seed: 347, replicateCount: REPLICATE_COUNT},
    build: (graph, uploads) => buildInterpolationLens(kind, graph, uploads)
  });
}
