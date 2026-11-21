// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createGTWRIndexedSummaryNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-indexed-summary';
import {GPUGeographicallyTemporallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY as SUMMARY} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import * as groupedReduction from '../../../src/gpu-spatial-analysis/geographic-distribution/grouped-sum-reduction';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

for (const route of ['dense', 'indexed'] as const) {
  for (const exact of [false, true]) {
    it(`${route} GTWR publishes ${exact ? 'exact-fit criteria' : 'usable likelihood separately from unavailable AICc'}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const rows = exact ? 8 : 4;
        const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
          name: string,
          format: Format,
          length: number
        ) => createTransientView(graph, name, format, length);
        const props = {
          positions: view('positions', 'float32x2', rows),
          times: view('times', 'float32', rows),
          temporalContext: {
            unit: 'second' as const,
            epoch: 0n,
            encoding: 'float32-relative' as const,
            axis: {kind: 'irregular' as const}
          },
          predictors: view('predictors', 'float32', rows),
          predictorCount: 1,
          response: view('response', 'float32', rows),
          parameters: view('parameters', 'float32', 4),
          output: {
            coefficients: view('coefficients', 'float32', rows * 2),
            fitted: view('fitted', 'float32', rows),
            hatDiagonal: view('hat', 'float32', rows),
            localStatus: view('status', 'uint32', rows),
            summary: view('summary', 'float32', 12),
            residuals: view('residuals', 'float32', rows)
          }
        };
        const memory = createCPUGraphMemory();
        let nodes;
        {
          // The reductions have cooperative barriers. Exercise their actual publication body
          // with independently hand-reduced scalar inputs, rather than emulating barriers.
          const original = groupedReduction.getGroupedSumNodes;
          const reduction = vi
            .spyOn(groupedReduction, 'getGroupedSumNodes')
            .mockImplementation((targetGraph, reductionProps) => {
              if (!reductionProps.finish) return [];
              for (const binding of reductionProps.finish.bindings) {
                const value = {
                  residualSums: exact ? 0 : 4,
                  hatTrace: 2,
                  observationCount: rows,
                  includedCount: rows
                }[
                  binding.name as 'residualSums' | 'hatTrace' | 'observationCount' | 'includedCount'
                ];
                if (value !== undefined)
                  memory.write(
                    binding.view,
                    binding.type === 'u32' ? new Uint32Array([value]) : new Float32Array([value])
                  );
              }
              return [
                createWGSLKernelNode(targetGraph, {
                  id: 'indexed-publication',
                  operation: 'SummaryPublicationTest',
                  invocationCount: 1,
                  bindings: reductionProps.finish.bindings,
                  declarations: reductionProps.finish.declarations,
                  // Sum (y-mean)^2 for y=1..n: n(n^2-1)/12.
                  body: `let sum0 = ${(rows * (rows * rows - 1)) / 12}.0;\n${reductionProps.finish.body}`
                })
              ];
            });
          try {
            nodes = (
              route === 'indexed'
                ? createGTWRIndexedSummaryNodes(graph, 'criteria', props, props.output.residuals)
                : new GPUGeographicallyTemporallyWeightedRegression({
                    ...props,
                    id: 'criteria',
                    searchStructure: 'brute-force'
                  }).getCommandNodes(graph)
            ).filter(node => node.id === 'indexed-publication');
          } finally {
            reduction.mockRestore();
          }
          expect(groupedReduction.getGroupedSumNodes).toBe(original);
        }
        const response = Float32Array.from({length: rows}, (_, row) => row + 1);
        memory.write(props.response, response);
        memory.write(
          props.output.fitted,
          exact ? response : Float32Array.from(response, value => value - 1)
        );
        memory.write(props.output.hatDiagonal, new Float32Array(rows).fill(2 / rows));
        memory.write(props.output.localStatus, new Uint32Array(rows));
        memory.write(props.output.summary, new Float32Array(12).fill(123));
        executeCommandNodesOnCPU(nodes, memory);
        const summary = memory.readFloat32(props.output.summary);
        expect(summary[SUMMARY.OBSERVATION_COUNT]).toBe(rows);
        expect(summary[SUMMARY.INCLUDED_ROW_COUNT]).toBe(rows);
        expect(summary[SUMMARY.SINGULAR_ROW_COUNT]).toBe(0);
        expect(Number.isNaN(summary[SUMMARY.HAS_VALID_CANDIDATE])).toBe(true);
        if (exact) {
          expect(summary[SUMMARY.LOG_LIKELIHOOD]).toBe(Infinity);
          expect(summary[SUMMARY.AICC]).toBe(-Infinity);
        } else {
          // n=4, RSS=4: llf=-n/2*(log(RSS/n)+log(2*pi)+1).
          // trace=2 makes n-trace-2=0, so AICc is unavailable independently of llf.
          expect(Number.isFinite(summary[SUMMARY.LOG_LIKELIHOOD])).toBe(true);
          expect(summary[SUMMARY.LOG_LIKELIHOOD]).toBeCloseTo(-5.675754132818691, 5);
          expect(Number.isNaN(summary[SUMMARY.AICC])).toBe(true);
        }
      } finally {
        device.destroy();
      }
    });
  }
}
