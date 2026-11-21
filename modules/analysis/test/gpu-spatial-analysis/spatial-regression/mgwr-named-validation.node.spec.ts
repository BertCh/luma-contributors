// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const [port, required] of [
  ['coefficients', 6],
  ['fitted', 3]
] as const) {
  for (const delta of [-1, 1]) {
    it(`names output.${port} and its exact size for length ${required + delta}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const output = {
          coefficients: createTransientView(
            graph,
            'coefficients',
            'float32',
            port === 'coefficients' ? required + delta : 6
          ),
          fitted: createTransientView(
            graph,
            'fitted',
            'float32',
            port === 'fitted' ? required + delta : 3
          ),
          status: {
            codes: createTransientView(graph, 'codes', 'uint32', 3),
            iterationCount: createTransientView(graph, 'iterations', 'uint32', 1)
          }
        };
        expect(
          () =>
            new GPUMultiscaleGeographicallyWeightedRegression({
              id: 'mgwr-validation',
              positions: createTransientView(graph, 'positions', 'float32x2', 3),
              predictors: createTransientView(graph, 'predictors', 'float32', 3),
              response: createTransientView(graph, 'response', 'float32', 3),
              parameters: createTransientView(graph, 'parameters', 'float32', 3),
              predictorCount: 1,
              output
            })
        ).toThrow(
          new RegExp(`mgwr-validation.*output\\.${port}\\.length.*\\[${required}, ${required}\\]`)
        );
      } finally {
        device.destroy();
      }
    });
  }
}

for (const [port, required] of [
  ['standardErrors', 6],
  ['summary', 4]
] as const) {
  it(`names inference.output.${port} and its exact size`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      expect(
        () =>
          new GPUMultiscaleGeographicallyWeightedRegression({
            id: 'mgwr-inference-validation',
            positions: createTransientView(graph, 'positions', 'float32x2', 3),
            predictors: createTransientView(graph, 'predictors', 'float32', 3),
            response: createTransientView(graph, 'response', 'float32', 3),
            parameters: createTransientView(graph, 'parameters', 'float32', 3),
            predictorCount: 1,
            inference: {
              output: {
                standardErrors: createTransientView(
                  graph,
                  'standard-errors',
                  'float32',
                  port === 'standardErrors' ? 5 : 6
                ),
                effectiveParameters: createTransientView(
                  graph,
                  'effective-parameters',
                  'float32',
                  2
                ),
                summary: createTransientView(
                  graph,
                  'inference-summary',
                  'float32',
                  port === 'summary' ? 3 : 4
                ),
                status: createTransientView(graph, 'inference-status', 'uint32', 1)
              }
            },
            output: {
              coefficients: createTransientView(graph, 'coefficients', 'float32', 6),
              fitted: createTransientView(graph, 'fitted', 'float32', 3),
              status: {
                codes: createTransientView(graph, 'codes', 'uint32', 3),
                iterationCount: createTransientView(graph, 'iterations', 'uint32', 1)
              }
            }
          })
      ).toThrow(
        new RegExp(
          `mgwr-inference-validation.*inference\\.output\\.${port}\\.length.*when inference is requested.*\\[${required}, ${required}\\]`
        )
      );
    } finally {
      device.destroy();
    }
  });
}

for (const [rows, maximumIterations, expected] of [
  [129, 1, /response\.length.*when inference is requested.*128/],
  // 128³ × (1+1) × 128 = 536870912 > 268435456; the row limit alone accepts 128.
  [128, 128, /inference work.*when inference is requested.*268435456/]
] as const) {
  it(`enforces shared full-inference bounds at ${rows} rows and ${maximumIterations} sweeps`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `inference-limit-${serial++}`, format, length);
      expect(
        () =>
          new GPUMultiscaleGeographicallyWeightedRegression({
            id: 'mgwr-inference-limits',
            positions: view('float32x2', rows),
            predictors: view('float32', rows),
            response: view('float32', rows),
            predictorCount: 1,
            parameters: view('float32', 3),
            maximumIterations,
            output: {
              coefficients: view('float32', rows * 2),
              fitted: view('float32', rows),
              status: {codes: view('uint32', rows), iterationCount: view('uint32', 1)}
            },
            inference: {
              output: {
                standardErrors: view('float32', rows * 2),
                effectiveParameters: view('float32', 2),
                summary: view('float32', 4),
                status: view('uint32', 1)
              }
            }
          })
      ).toThrow(expected);
    } finally {
      device.destroy();
    }
  });
}
