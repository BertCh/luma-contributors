// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const mode of ['tiled', 'reference'] as const) {
  it(`B3 rejects unsupported packed, root and child weights evidence in ${mode} mode`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const packed = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6)
    };
    const evidence = createTransientView(graph, 'evidence', 'uint32', 1);
    const chunk = {rowStart: 0, rowCount: 4, weights: packed};
    const chunked = {rowCount: 4, chunks: [chunk]};
    const props = {
      id: 'global-evidence',
      mode,
      values: createTransientView(graph, 'values', 'float32', 4),
      statistics: ['moran'] as const,
      output: {summary: createTransientView(graph, 'summary', 'float32', 52)}
    };
    try {
      for (const field of [
        'requiredCount',
        'converged',
        'iterationLimitReached',
        'unresolvedCount'
      ] as const) {
        const status = {[field]: evidence};
        for (const weights of [
          {...packed, status},
          {...chunked, status}
        ]) {
          expect(() => new GPUGlobalSpatialStatistics({...props, weights})).toThrow(
            new RegExp(`global-evidence.*weights\\.status\\.${field}`)
          );
        }
        const weights = {...chunked, chunks: [{...chunk, weights: {...packed, status}}]};
        expect(() => new GPUGlobalSpatialStatistics({...props, weights})).toThrow(
          new RegExp(`global-evidence.*weights\\.chunks\\[0\\]\\.weights\\.status\\.${field}`)
        );
      }
    } finally {
      device.destroy();
    }
  });

  it(`S1 requires one-word chunk-root loss views in ${mode} mode`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'loss-backing', 'uint32', 3);
    const scalar = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    const weights = {
      rowCount: 4,
      chunks: [
        {
          rowStart: 0,
          rowCount: 4,
          weights: {
            offsets: createTransientView(graph, 'offsets', 'uint32', 5),
            neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
            weights: createTransientView(graph, 'weights', 'float32', 6)
          }
        }
      ]
    };
    const values = createTransientView(graph, 'values', 'float32', 4);
    const summary = createTransientView(graph, 'summary', 'float32', 52);
    const construct = (status: {overflow?: typeof scalar; candidateOverflow?: typeof scalar}) =>
      new GPUGlobalSpatialStatistics({
        id: 'global-scalar',
        mode,
        weights: {...weights, status},
        values,
        statistics: ['moran'],
        output: {summary}
      });
    try {
      for (const field of ['overflow', 'candidateOverflow'] as const) {
        expect(() => construct({[field]: backing})).toThrow(
          new RegExp(`global-scalar weights\\.status ${field} must contain one uint32 row`)
        );
        expect(() => construct({[field]: scalar})).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });
}

for (const mode of ['tiled', 'reference'] as const)
  for (const location of ['whole', 'root', 'child'] as const)
    for (const clearFlags of [false, true])
      it(`W5-18 rejects a preserved ${location} complete certificate before graph allocation (${mode}, flags ${clearFlags})`, () => {
        const device = createNullWebGPUDevice();
        try {
          for (const certificate of [0, 1]) {
            const graph = new GPUCommandGraph(device);
            let serial = 0;
            const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
              createTransientView(graph, `view-${serial++}`, format, length);
            const buffer = device.createBuffer({
              data: Uint32Array.of(certificate),
              usage: Buffer.STORAGE
            });
            try {
              const complete = importGraphBuffer(
                graph,
                'complete-certificate',
                buffer,
                'uint32',
                1
              );
              // Prototype accessors exercise capture as well as the collector's inverse-polarity
              // guard. Zero is proof of loss, even alongside two clear loss flags. This consumer
              // explicitly refuses all certificates, so neither 0 nor 1 can be silently ignored.
              const status = Object.assign(
                Object.create({
                  get complete() {
                    return complete;
                  }
                }),
                clearFlags
                  ? {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
                  : {}
              );
              const child = {
                offsets: view('uint32', 5),
                neighbors: view('uint32', 6),
                weights: view('float32', 6),
                ...(location === 'child' ? {status} : {})
              };
              const weights =
                location === 'whole'
                  ? {...child, status}
                  : {
                      rowCount: 4,
                      chunks: [{rowStart: 0, rowCount: 4, weights: child}],
                      ...(location === 'root' ? {status} : {})
                    };
              const props = {
                id: 'certificate-consumer',
                mode,
                weights,
                values: view('float32', 4),
                statistics: ['geary'] as const,
                output: {
                  summary: view('float32', 52),
                  status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
                }
              };
              const allocate = vi.spyOn(graph, 'createTransientBuffer');
              try {
                expect(() => new GPUGlobalSpatialStatistics(props)).toThrow(
                  /certificate-consumer.*weights\.status\.complete.*completeness-aware/
                );
                expect(allocate).not.toHaveBeenCalled();
              } finally {
                allocate.mockRestore();
              }
            } finally {
              buffer.destroy();
            }
          }
        } finally {
          device.destroy();
        }
      });
