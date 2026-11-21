// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import {GPUCommandGraph, GraphVectorView, createTransientView} from '@luma.gl/graph/gpu-core';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {decodeInterpolationPredictionStatus} from '../../../src/gpu-spatial-analysis/spatial-interpolation/interpolation-neighborhood-loss';
import {
  GPUNaturalNeighbor,
  GPULocalKriging,
  GPULocalRBF,
  GPUConditionalGaussianSimulation
} from '@luma-contributors/analysis/gpu-spatial-analysis';

it('F2/F4 prediction status accepts foreign words and offset bytes and rejects wrong kinds', () => {
  for (const expression of [
    'Uint32Array.of(0, 3)',
    'new Uint8Array(Uint32Array.of(91, 0, 3, 92).buffer, 4, 8)',
    'Uint32Array.of(0, 3).buffer'
  ]) {
    const input = runInNewContext(expression) as Uint32Array | Uint8Array | ArrayBuffer;
    expect(decodeInterpolationPredictionStatus('prediction', input, 1)).toEqual({
      code: 3,
      complete: 0,
      invalidInput: 0,
      singular: 0,
      candidateOverflow: 1
    });
  }
  for (const input of [Float32Array.of(3), new DataView(Uint32Array.of(3).buffer)]) {
    expect(() => {
      // @ts-expect-error Exercise runtime validation of unsupported readback types.
      decodeInterpolationPredictionStatus('prediction', input);
    }).toThrow(/status must be/);
  }
  const disguisedFloats = Float32Array.of(3);
  Object.defineProperty(disguisedFloats, Symbol.toStringTag, {value: 'ArrayBuffer'});
  expect(() =>
    decodeInterpolationPredictionStatus('prediction', disguisedFloats as unknown as Uint32Array)
  ).toThrow(/status must be/);
  expect(() => decodeInterpolationPredictionStatus('prediction', new Uint8Array(3))).toThrow(
    /whole/
  );
  expect(() => decodeInterpolationPredictionStatus('prediction', Uint32Array.of(0), 1)).toThrow(
    /row 1/
  );
});

it('B1 public interpolation constructors retain caller, class and indexed port paths', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packed = createTransientView(graph, 'first', 'float32x2', 2);
    const storage = createTransientView(graph, 'second', 'float32x2', 4);
    const strided = graph.createDataView(storage.buffer, {
      format: 'float32x2',
      length: 2,
      byteStride: 16
    });
    const chunks = new GraphVectorView({
      id: 'chunks',
      name: 'chunks',
      format: 'float32x2',
      length: 4,
      valueLength: 8,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: [packed, strided]
    });
    const base = {
      id: 'test west',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      output: {
        values: createTransientView(graph, 'output-values', 'float32x4', 4),
        status: createTransientView(graph, 'output-status', 'uint32', 4)
      }
    };
    const samples = createTransientView(graph, 'samples', 'float32', 3);
    const observations = createTransientView(graph, 'observations', 'float32x4', 3);
    const cases: [string, string, () => unknown][] = [
      [
        'GPUNaturalNeighbor',
        'queries chunk 1',
        () =>
          new GPUNaturalNeighbor({
            ...base,
            queries: chunks,
            values: samples,
            maximumNeighborCount: 3
          })
      ],
      [
        'GPULocalKriging',
        'supports chunk 1',
        () =>
          new GPULocalKriging({
            ...base,
            supports: chunks,
            mode: 'ordinary',
            neighborCount: 3,
            observations,
            covariance: {model: 'exponential', range: 1, sill: 1}
          })
      ],
      [
        'GPULocalRBF',
        'queries chunk 1',
        () => new GPULocalRBF({...base, queries: chunks, values: samples, neighborCount: 3})
      ],
      [
        'GPUConditionalGaussianSimulation',
        'positions chunk 0',
        () =>
          new GPUConditionalGaussianSimulation({
            id: base.id,
            positions: strided,
            observations,
            queries: packed,
            covariance: {model: 'exponential', range: 1, sill: 1},
            seed: 1,
            replicateCount: 1,
            output: {values: samples, moments: packed, status: base.output.status}
          })
      ]
    ];
    for (const [contributor, path, construct] of cases) {
      let caught: unknown;
      try {
        construct();
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(AnalysisValidationError);
      const failure = caught as AnalysisValidationError;
      expect({id: failure.id, contributor: failure.contributor, path: failure.path}).toEqual({
        id: 'test west',
        contributor,
        path
      });
    }
  } finally {
    device.destroy();
  }
});
