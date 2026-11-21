// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSinkhornDivergence} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('retains the one-round two-cell iterate with default and explicit stopping destinations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const explicit of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const source = createInputBuffer(device, Float32Array.of(9, 1));
    const target = createInputBuffer(device, Float32Array.of(1, 9));
    const parameters = createInputBuffer(
      device,
      getGPUSinkhornDivergenceParameterValues({cellWidth: 1, cellHeight: 1, epsilon: 0.5})
    );
    const divergence = createOutputBuffer(device, 1);
    const error = createOutputBuffer(device, 1);
    const converged = createOutputBuffer(device, 1);
    const limited = createOutputBuffer(device, 1);
    const iterations = createOutputBuffer(device, 1);
    converged.write(Uint32Array.of(77));
    const errorView = importGraphBuffer(graph, 'error', error, 'float32', 1);
    const convergedView = importGraphBuffer(graph, 'converged', converged, 'uint32', 1);
    const limitedView = importGraphBuffer(graph, 'limited', limited, 'uint32', 1);
    const iterationsView = importGraphBuffer(graph, 'iterations', iterations, 'uint32', 1);
    const solver = new GPUSinkhornDivergence({
      source: importGraphBuffer(graph, 'source', source, 'float32', 2),
      target: importGraphBuffer(graph, 'target', target, 'float32', 2),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 6),
      grid: {width: 2, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, 1, 0]},
      maximumIterations: 1,
      output: {
        divergence: importGraphBuffer(graph, 'divergence', divergence, 'float32', 1),
        ...(explicit
          ? {
              marginalError: errorView,
              status: {
                converged: convergedView,
                iterationLimitReached: limitedView,
                iterationCount: iterationsView
              }
            }
          : {})
      }
    });
    graph.add(solver);
    if (!explicit) {
      for (const [name, input, output, type] of [
        ['error', solver.output.marginalError!, errorView, 'f32'],
        ['converged', solver.output.status!.converged!, convergedView, 'u32'],
        ['limited', solver.output.status!.iterationLimitReached!, limitedView, 'u32'],
        ['iterations', solver.output.status!.iterationCount!, iterationsView, 'u32']
      ] as const)
        graph.add(
          createWGSLKernelNode(graph, {
            id: `copy-${name}`,
            operation: 'stopping-test',
            variant: 'copy',
            invocationCount: 1,
            bindings: [
              {name: 'input', view: input, type, access: 'read'},
              {name: 'output', view: output, type, access: 'read_write'}
            ],
            body: 'output[outputOffset] = input[inputOffset];'
          })
        );
    } else {
      expect(solver.output.status!.converged).toBe(convergedView);
    }
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    // Independent matrix algebra: K=[[1,exp(-2)],[exp(-2),1]], a=(.9,.1), b=(.1,.9).
    // g=-log(K a)/2, f=-log(K (b exp(2g)))/2, selfA=-log(K a)/4, selfB=-log(K b)/4.
    // S=a·(f-selfA)+b·(g-selfB); error=sum(abs(b*(K a)-b)).
    const value = (await readFloat32(divergence, 1))[0];
    const residual = (await readFloat32(error, 1))[0];
    expect(Number.isFinite(value)).toBe(true);
    expect(Number.isFinite(residual)).toBe(true);
    expect(value).toBeCloseTo(0.6839309931, 5);
    expect(residual).toBeCloseTo(0.709025085, 5);
    expect(await readUint32(converged, 1)).toEqual([0]);
    expect(await readUint32(limited, 1)).toEqual([1]);
    expect(await readUint32(iterations, 1)).toEqual([1]);
    compiled.destroy();
    for (const buffer of [
      source,
      target,
      parameters,
      divergence,
      error,
      converged,
      limited,
      iterations
    ])
      buffer.destroy();
  }
});
