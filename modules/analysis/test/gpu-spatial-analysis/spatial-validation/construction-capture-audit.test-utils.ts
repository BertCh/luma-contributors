// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {NestedInterpolationValidationOperationProps} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';

/** Inherited, nonenumerable getters expose repeated reads and uncaptured nested maps. */
export function watchConstructionFields<Props extends object>(values: Props, label: string) {
  const reads = new Map<string, number>();
  const prototype = Object.create(null);
  for (const field of Object.keys(values)) {
    Object.defineProperty(prototype, field, {
      get() {
        reads.set(field, (reads.get(field) ?? 0) + 1);
        return values[field as keyof Props];
      }
    });
  }
  return {
    props: Object.create(prototype) as Props,
    expectReadOnce() {
      for (const field of Object.keys(values))
        expect(reads.get(field), `${label}.${field}`).toBe(1);
    }
  };
}

/** Hand oracle: two ordinary-kriging neighbors have independent unit covariance. */
export function createNestedCaptureInput(scale = 1) {
  // Six constant training outcomes imply weights [1/2,1/2] and prediction = scale.
  // Test outcomes [2,4]*scale give SSE=10*scale², SST=2*scale² and R²=-4.
  return {
    positions: Float32Array.from([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0]),
    response: Float32Array.from([1, 1, 1, 1, 1, 1, 2, 4], value => value * scale),
    training: Uint32Array.from([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]),
    testing: Uint32Array.from([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
    innerBlocks: Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7])
  };
}

export function createNestedCaptureProps(
  mode: 'ordinary-kriging' | 'rbf' = 'ordinary-kriging'
): NestedInterpolationValidationOperationProps {
  return {
    id: 'captured-nested',
    mode: 'tiled',
    rowCount: 8,
    positions: 'positions',
    response: 'response',
    training: 'training',
    testing: 'testing',
    innerBlocks: 'innerBlocks',
    foldCount: 2,
    innerFoldCount: 2,
    innerBufferDistance: 0,
    seed: 0,
    neighborCount: 2,
    candidates:
      mode === 'ordinary-kriging'
        ? [
            {
              mode: 'ordinary-kriging',
              covariance: {model: 'spherical', range: 0.5, sill: 1, nugget: 0}
            }
          ]
        : [{mode: 'rbf', kernel: 'gaussian', degree: 0, epsilon: 0.5, smoothing: 0.25}],
    output: {
      selected: 'selected',
      scores: 'scores',
      predictions: 'predictions',
      metrics: 'metrics',
      status: 'status'
    }
  };
}

export function getNestedCaptureExpected(scale = 1) {
  return {
    selected: [0, 0],
    scores: [0, 6, 0, 0, 6, 0],
    predictions: [
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      scale,
      scale,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      scale,
      scale
    ],
    metrics: [
      2,
      0,
      5 * scale * scale,
      2 * scale,
      -2 * scale,
      -4,
      2,
      0,
      5 * scale * scale,
      2 * scale,
      -2 * scale,
      -4
    ],
    status: [0, 0]
  };
}
