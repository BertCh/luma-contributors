// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH,
  GPURasterZonalStatistics
} from '../../../src/raster-analysis/raster-zonal-statistics/index';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

// The float32 two-pass variance around a float32 mean is badly wrong for values near 1e7 over many
// cells (a float32 model gives 123 instead of 2.00002 for 100,000 cells; see
// raster-zonal-variance-precision.spec.ts). The variance subgraph must refine the mean with a
// deviation pass and subtract the corrected two-pass term, in both sum orders and with weights.
for (const sumOrder of ['sorted', 'atomic'] as const) {
  for (const weighted of [false, true]) {
    it(`GPURasterZonalStatistics variances refine the mean and use the corrected two-pass sum (${sumOrder}${weighted ? ', geographic' : ''})`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const grid = makeGPURasterMetadata({width: 4, height: 3});
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const nodes = new GPURasterZonalStatistics({
        id: 'zonal',
        grid,
        zones: view('zones', 'uint32', 12),
        values: {
          id: 'values',
          format: 'float32',
          storage: {kind: 'buffer', values: view('values', 'float32', 12)}
        },
        zoneCapacity: 2,
        sumOrder,
        ...(weighted
          ? {
              cellSizeMode: 'geographic' as const,
              parameters: view(
                'parameters',
                'float32',
                GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH
              )
            }
          : {}),
        output: {
          variances: view('variances', 'float32', 2),
          standardDeviations: view('deviations', 'float32', 2)
        }
      }).getCommandNodes(graph);
      const ids = nodes.map(node => node.id);
      const order = [
        'zonal-means',
        'zonal-first-deviations',
        'zonal-refined-means',
        'zonal-deviations',
        'zonal-variances'
      ];
      const positions = order.map(id => ids.indexOf(id));
      expect(
        positions.every(position => position >= 0),
        ids.join(',')
      ).toBe(true);
      expect(positions).toEqual([...positions].sort((left, right) => left - right));
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of sources) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
      const variance = sources.find(kernel => kernel.id.endsWith('zonal-variances'))!;
      expect(variance.source).toMatch(/linearDeviationSum \/ total\) \* linearDeviationSum/);
      const refined = sources.find(kernel => kernel.id.endsWith('zonal-refined-means'))!;
      expect(refined.source).toContain(
        'linearDeviationSums[linearDeviationSumsOffset + index] / total'
      );
    });
  }
}
