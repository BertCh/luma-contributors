// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const period of ['before', 'after'] as const) {
  for (const reportSumLoss of [false, true]) {
    it(`comparison consumes ${period}-only saturation with optional output flag ${reportSumLoss}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format) =>
        createTransientView(graph, `sum-loss-${serial++}`, format, 1);
      const table = (name: 'before' | 'after') => ({
        cells: view('uint32x2'),
        counts: view('uint32'),
        sums: view('uint32x2'),
        status: {
          count: view('uint32'),
          overflow: view('uint32'),
          ...(name === period ? {sumOverflow: view('uint32')} : {})
        }
      });
      try {
        const nodes = new GPUCellTableCompare({
          id: 'sum-loss',
          measure: 'sum',
          before: table('before'),
          after: table('after'),
          output: {
            cells: view('uint32x2'),
            presence: view('uint32'),
            delta: view('float32'),
            zScore: view('float32'),
            status: {
              count: view('uint32'),
              overflow: view('uint32'),
              candidateOverflow: view('uint32'),
              ...(reportSumLoss ? {sumOverflow: view('uint32')} : {})
            }
          }
        }).getCommandNodes(graph);
        expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
        const sources = getGPUCommandNodeKernelSources(nodes);
        const suppress = sources.find(kernel => kernel.id === 'sum-loss-sum-loss');
        expect(suppress?.source).toContain('sumIn0');
        const suppressions = sources.filter(
          kernel => kernel.id === 'sum-loss-sum-loss' || kernel.id === 'sum-loss-sum-loss-z-score'
        );
        expect(suppressions).toHaveLength(2);
        for (const kernel of suppressions) {
          // A direct literal bitcast is a non-finite WGSL constant expression, rejected by
          // shader creation even when the runtime overflow flag is zero (review B1).
          expect(kernel.source).not.toMatch(/bitcast<f32>\(\s*0x7fc00000u\s*\)/);
          expect(kernel.source).toContain('var bits = 0x7fc00000u;');
          expect(kernel.source).toContain('getNaN()');
          // Only occupied rows may lose sum measures; empty before/after rows stay zero (B2).
          expect(kernel.source).toContain('index < total[totalOffset]');
        }
        expect(sources.some(kernel => kernel.id === 'sum-loss-sum-status')).toBe(reportSumLoss);
        const candidate = sources.find(kernel => kernel.id === 'sum-loss-candidate-status');
        expect(candidate?.source).not.toContain('sumIn');
      } finally {
        device.destroy();
      }
    });
  }
}
