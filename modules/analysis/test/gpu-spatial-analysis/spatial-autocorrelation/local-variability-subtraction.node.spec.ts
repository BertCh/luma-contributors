// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPULocalSpatialHeteroscedasticity} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import * as rowDispatch from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-row-dispatch';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

for (const chunked of [false, true])
  for (const residualPower of [1, 2] as const)
    for (const correctionLoss of [false, true])
      it(`W5-16 ${correctionLoss ? 'B1 correction sum' : 'original subtraction'} LOSH retains subtraction errors (${chunked ? 'chunked' : 'whole'}, power ${residualPower})`, () => {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device);
        const memory = createCPUGraphMemory();
        let serial = 0;
        const input = <Format extends 'float32' | 'uint32'>(
          values: Float32Array | Uint32Array,
          format: Format
        ) => {
          const view = createTransientView(graph, `input-${serial++}`, format, values.length);
          memory.write(view, values);
          return view;
        };
        const d = 2 ** -25;
        const u = 1 - 2 ** -24;
        const initialValues = correctionLoss
          ? [0.25 + d, 1, 0.25, -0.5 + 2 * d, 0.25 + 2 * d]
          : [2 ** -26, 2 ** -25, 1, 2 ** -25, -1];
        const initialWeights = correctionLoss ? [1, u, 1, u] : [1, 1, 1, 1];
        const edgeWeights = input(
          Float32Array.from([...initialWeights, ...(chunked ? [] : [1, 1, 1, 1])]),
          'float32'
        );
        const values = input(Float32Array.from(initialValues), 'float32');
        const first = {
          offsets: input(Uint32Array.of(0, 4), 'uint32'),
          neighbors: input(Uint32Array.of(1, 2, 3, 4), 'uint32'),
          weights: edgeWeights
        };
        const weights = chunked
          ? {
              rowCount: 5,
              chunks: [
                {rowStart: 0, rowCount: 1, weights: first},
                {
                  rowStart: 1,
                  rowCount: 4,
                  weights: {
                    offsets: input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
                    neighbors: input(Uint32Array.of(0, 0, 0, 0), 'uint32'),
                    weights: input(Float32Array.of(1, 1, 1, 1), 'float32')
                  }
                }
              ]
            }
          : {
              offsets: input(Uint32Array.of(0, 4, 5, 6, 7, 8), 'uint32'),
              neighbors: input(Uint32Array.of(1, 2, 3, 4, 0, 0, 0, 0), 'uint32'),
              weights: edgeWeights
            };
        const allocations = vi.spyOn(graph, 'createTransientBuffer');
        // The interpreter accepts storage arrays but not uniform structs. Specialize only the
        // dispatch bounds; the producer's declarations, bindings and residual body run unchanged.
        const dispatch = vi
          .spyOn(rowDispatch, 'createSpatialStatisticsRowNode')
          .mockImplementation((target, props) =>
            createWGSLKernelNode(target, {
              ...props,
              declarations: `const CAPACITY: u32 = ${props.bindings.find(binding => binding.name === 'neighbors')!.view.length}u;
const WEIGHTS_ROW_START: u32 = ${props.weightsRowStart ?? props.rowBase ?? 0}u;\n${props.declarations ?? ''}`,
              body: `{ let index = index + ${props.rowBase ?? 0}u;\n${props.body}\n}`
            })
          );
        try {
          const nodes = new GPULocalSpatialHeteroscedasticity({
            id: 'subtraction',
            weights,
            values,
            residualPower,
            output: {
              statistic: createTransientView(graph, 'statistic', 'float32', 5),
              status: createTransientView(graph, 'status', 'uint32', 5)
            }
          }).getCommandNodes(graph);
          const residualBuffer = allocations.mock.results
            .map(result => result.value)
            .find(buffer => buffer.id === 'subtraction-residuals');
          expect(residualBuffer).toBeDefined();
          const residuals = graph.createDataView(residualBuffer, {format: 'float32', length: 5});
          // Execute the actual producer through the raw residual pass. Parallel reductions later
          // in the graph are deliberately outside this interpreter's supported kernel set.
          const end = nodes.findIndex(node => node.id === 'subtraction-finite-residuals');
          expect(end).toBeGreaterThan(0);
          const residualNodes = nodes.slice(0, end);
          executeCommandNodesOnCPU(residualNodes, memory);
          // Exact arithmetic: (2d + 1 + 2d - 1)/4 = d, so focus d is harmonic.
          expect(memory.readFloat32(residuals)[0]).toBe(0);
          // Same neighbors, focus 0: a genuinely nonzero residual d**power, not a zero clamp.
          if (correctionLoss) {
            // Equal-weight pairs average exactly to the focus. Changing only weight[1] from u
            // to 1 gives weighted difference -d*(1-u)=-2^-49, total weight 4-2^-24.
            memory.write(
              edgeWeights,
              Float32Array.from([1, 1, 1, u, ...(chunked ? [] : [1, 1, 1, 1])])
            );
          } else memory.write(values, Float32Array.of(0, 2 ** -25, 1, 2 ** -25, -1));
          executeCommandNodesOnCPU(residualNodes, memory);
          const expected = correctionLoss
            ? Math.fround((2 ** -49 / (4 - 2 ** -24)) ** residualPower)
            : 2 ** (-26 * residualPower);
          const actual = memory.readFloat32(residuals)[0];
          expect(Number.isFinite(actual)).toBe(true);
          expect(actual).toBeGreaterThan(0);
          expect(Math.abs(actual / expected - 1)).toBeLessThan(2e-6);
        } finally {
          allocations.mockRestore();
          dispatch.mockRestore();
          device.destroy();
        }
      });
