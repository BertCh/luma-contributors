// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

for (const mode of ['tiled', 'reference'] as const) {
  for (const chunked of [false, true]) {
    it(`global ${mode} statistics suppress candidate-only loss with ${chunked ? 'chunked' : 'packed'} weights`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `global-loss-${serial++}`, format, length);
      const weights = {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 6),
        weights: view('float32', 6)
      };
      const status = {candidateOverflow: view('uint32', 1)};
      const props = {
        id: 'global-loss',
        mode,
        weights: chunked
          ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}], status}
          : {...weights, status},
        values: view('float32', 4),
        statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'] as const,
        secondValues: view('float32', 4),
        output: {
          summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
          joinCounts: view('uint32', 3),
          counts: view('uint32', 3),
          status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
        }
      };
      try {
        const sources = getGPUCommandNodeKernelSources(
          new GPUGlobalSpatialStatistics(props).getCommandNodes(graph)
        );
        const publisher = sources.find(record => record.id === 'global-loss-weights-status');
        expect(
          publisher,
          'the input candidate flag must reach the aggregate suppression flag'
        ).toBeDefined();
        expect(publisher!.source).toContain(
          'inputCandidateOverflow[inputCandidateOverflowOffset] != 0u'
        );
        expect(publisher!.source).toContain(
          'incomplete[incompleteOffset] = finalLoss | candidateLoss'
        );
        expect(publisher!.source).toContain(
          'candidateOverflow[candidateOverflowOffset] = candidateLoss'
        );
        expect(publisher!.source).toContain('overflow[overflowOffset] = finalLoss');
        const finalize = sources.find(
          record => record.id === `global-loss-${mode === 'tiled' ? 'finalize-tiled' : 'finalize'}`
        )!;
        expect(finalize.source).toContain('overflow[overflowOffset] != 0u');
        for (const record of sources) {
          expect(getStorageBindingCount(record.source), record.id).toBeLessThanOrEqual(8);
        }
        const bothFlags = {
          ...props.weights,
          status: {overflow: view('uint32', 1), candidateOverflow: status.candidateOverflow}
        };
        const folded = getGPUCommandNodeKernelSources(
          new GPUGlobalSpatialStatistics({
            ...props,
            id: 'global-loss-both',
            weights: bothFlags
          }).getCommandNodes(graph)
        );
        const foldedPublisher = folded.find(
          record => record.id === 'global-loss-both-weights-status'
        )!;
        const reflection = new WgslReflect(foldedPublisher.source);
        const used = new Set(
          reflection.entry.compute.flatMap(entry => entry.resources.map(resource => resource.name))
        );
        expect(reflection.storage.every(binding => used.has(binding.name))).toBe(true);
        for (const record of folded)
          expect(getStorageBindingCount(record.source), record.id).toBeLessThanOrEqual(8);
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              ...props,
              output: {...props.output, status}
            })
        ).toThrow(/share buffers/);
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              ...props,
              output: {
                ...props.output,
                status: {
                  overflow: status.candidateOverflow,
                  candidateOverflow: status.candidateOverflow
                }
              }
            })
        ).toThrow(/overlap/);
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              ...props,
              weights: {...props.weights, status: {candidateOverflow: view('uint32', 0)}}
            })
        ).toThrow(/candidateOverflow/);
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              ...props,
              output: {...props.output, status: {candidateOverflow: view('uint32', 2)}}
            })
        ).toThrow(/candidateOverflow.*1/);
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              ...props,
              weights: {...props.weights, status: {candidateOverflow: view('float32', 1) as never}}
            })
        ).toThrow(/candidateOverflow/);

        const withoutInputs = getGPUCommandNodeKernelSources(
          new GPUGlobalSpatialStatistics({
            ...props,
            id: 'global-loss-zero',
            weights: {...props.weights, status: undefined}
          }).getCommandNodes(graph)
        );
        const zeroStatus = withoutInputs.find(
          record => record.id === 'global-loss-zero-weights-status'
        )!;
        expect(zeroStatus.source).toContain('let finalLoss = 0u');
        expect(zeroStatus.source).toContain('let candidateLoss = 0u');

        const foreignGraph = new GPUCommandGraph(device);
        const foreign = createTransientView(foreignGraph, 'foreign-global-loss', 'uint32', 1);
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        expect(() =>
          new GPUGlobalSpatialStatistics({
            ...props,
            weights: {...props.weights, status: {candidateOverflow: foreign}}
          }).getCommandNodes(graph)
        ).toThrow(/target graph/);
        expect(allocate).not.toHaveBeenCalled();
        allocate.mockRestore();
      } finally {
        device.destroy();
      }
    });
  }
}
