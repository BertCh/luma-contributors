// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {getGPUSpatialWeightsTilePlan} from '../../../src/gpu-spatial-analysis/spatial-regression/chunked-weights-kernels';
import * as contributorValidation from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import * as regressionTraceLayout from '../../../src/gpu-spatial-analysis/spatial-regression/regression-trace-layout';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const Contributor of [GPUSpatialLagMaximumLikelihood, GPUSpatialErrorMaximumLikelihood]) {
  for (const extraChunk of [false, true]) {
    it(`${Contributor.name} isolates nested traces at 4128 rows: extraChunk=${extraChunk}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        let serial = 0;
        const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
          createTransientView(graph, `trace-layout-${serial++}`, format, length);
        // Same total in both layouts: 4095 + 33 = 4096 + 32 = 4128.
        // The one-level target widens the first layout; the second uses hierarchical merging.
        const counts = [
          ...new Array<number>(extraChunk ? 4096 : 4095).fill(1),
          extraChunk ? 32 : 33
        ];
        let rowStart = 0;
        const chunks = counts.map(rowCount => {
          const chunk = {
            rowStart,
            rowCount,
            weights: {
              offsets: view('uint32', rowCount + 1),
              neighbors: view('uint32', rowCount),
              weights: view('float32', rowCount)
            }
          };
          rowStart += rowCount;
          return chunk;
        });
        expect(rowStart).toBe(4128);
        const weights = {rowCount: rowStart, chunks};
        expect(getGPUSpatialWeightsTilePlan(weights, 32).tileCount).toBe(4097);
        const errorMainTiles = getGPUSpatialWeightsTilePlan(weights, 64).tileCount;
        expect(errorMainTiles).toBe(extraChunk ? 4097 : 4096);
        const props = {
          id: 'isolated-likelihood',
          weights,
          tileRowCount: 64,
          predictorCount: 1,
          predictors: view('float32', rowStart),
          response: view('float32', rowStart),
          maximumPower: 2,
          traceMode: 'stochastic' as const,
          traceProbeCount: 1,
          output: {
            table: view('float32', 12),
            summary: view('float32', 32),
            status: {
              code: view('uint32', 1),
              converged: view('uint32', 1)
            }
          }
        };
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        expect(regressionTraceLayout.getRegressionTraceLayout(props.id, weights, rowStart)).toEqual(
          {
            tileRowCount: extraChunk ? 32 : 64,
            tileCount: extraChunk ? 4097 : 4096
          }
        );
        // Both trace layouts pass. Thousands of chunk dispatches independently exceed the
        // command budget; assert that real guard, then isolate it to inspect the real main plan.
        expect(() => new Contributor(props)).toThrow(/estimated command node count.*32768/);
        expect(allocate).not.toHaveBeenCalled();
        const commandBudget = vi
          .spyOn(contributorValidation, 'validateCommandNodeBudget')
          .mockImplementation(() => {});
        try {
          const contributor = new Contributor(props);
          expect(commandBudget).toHaveBeenCalledOnce();
          expect(Reflect.get(contributor, 'tileCount')).toBe(
            Contributor === GPUSpatialLagMaximumLikelihood ? 65 : extraChunk ? 4097 : 4096
          );
        } finally {
          commandBudget.mockRestore();
        }
      } finally {
        device.destroy();
      }
    });
  }
}
