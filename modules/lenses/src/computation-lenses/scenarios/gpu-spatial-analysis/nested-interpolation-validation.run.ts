// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {runExample} from './nested-interpolation-validation.example';
import {runS2Lens, type S2CaptureSpec} from './s2-lens-run';
export function runNestedInterpolationValidationLens(device: Device) {
  return runS2Lens(device, {
    scenarioId: 'gpu-spatial-analysis.nested-interpolation-validation',
    element: 'GPUNestedInterpolationValidation',
    datasets: ['synthetic-nested-validation-twelve-sites'],
    seed: 2,
    keySpaces: [
      {id: 'fold', label: 'Outer fold', size: 2, indexSpace: '1d'},
      {id: 'sample', label: 'Sample', size: 12, indexSpace: '1d'},
      {id: 'prediction', label: 'Fold-major predictions', size: 24, indexSpace: '1d'},
      {id: 'score', label: 'Fold/candidate score components', size: 12, indexSpace: '1d'},
      {id: 'metric', label: 'Heldout metric components', size: 12, indexSpace: '1d'}
    ],
    choices: {innerFoldCount: 2, neighborCount: 4},
    async build(graph, uploads) {
      const coordinates = Array.from({length: 12}, (_, row) => [
        row % 4,
        Math.floor(row / 4)
      ]).flat();
      const positions = uploads.view('positions', Float32Array.from(coordinates), 'float32x2');
      const response = uploads.view(
        'response',
        Float32Array.from(
          {length: 12},
          (_, row) => 2 + 0.3 * coordinates[row * 2] + 0.8 * coordinates[row * 2 + 1]
        ),
        'float32'
      );
      const training = uploads.view(
        'training',
        Uint32Array.from({length: 24}, (_, index) =>
          Number((index % 12) % 2 !== Math.floor(index / 12))
        ),
        'uint32'
      );
      const testing = uploads.view(
        'testing',
        Uint32Array.from({length: 24}, (_, index) =>
          Number((index % 12) % 2 === Math.floor(index / 12))
        ),
        'uint32'
      );
      const innerBlocks = uploads.view(
        'inner-blocks',
        Uint32Array.from({length: 12}, (_, row) => Math.floor(row / 2)),
        'uint32'
      );
      const captures: Record<string, S2CaptureSpec> = {
        response: {view: response, role: 'input', keySpace: 'sample'},
        positions: {
          view: positions,
          role: 'context',
          keySpace: 'sample',
          geometry: {
            kind: 'points',
            coordinateModel: 'projected',
            unit: 'native planar units',
            valuesCapture: 'response'
          }
        }
      };
      const output = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number,
        keySpace: string
      ) => {
        const view = createTransientView(
          graph,
          name,
          format,
          length,
          Buffer.STORAGE | Buffer.COPY_SRC
        );
        captures[name] = {view, role: 'output', keySpace};
        return view;
      };
      await runExample(graph, {
        positions,
        response,
        training,
        testing,
        innerBlocks,
        foldCount: 2,
        innerFoldCount: 2,
        neighborCount: 4,
        seed: 2,
        candidates: [
          {
            mode: 'ordinary-kriging',
            covariance: {model: 'exponential', range: 3, sill: 1, nugget: 0.01}
          },
          {mode: 'rbf', kernel: 'gaussian', epsilon: 0.5, smoothing: 0.1}
        ],
        output: {
          selected: output('selected', 'uint32', 2, 'fold'),
          scores: output('scores', 'float32', 12, 'score'),
          predictions: output('predictions', 'float32', 24, 'prediction'),
          metrics: output('metrics', 'float32', 12, 'metric'),
          status: output('status', 'uint32', 2, 'fold')
        }
      });
      return captures;
    }
  });
}
