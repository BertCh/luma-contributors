// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createPanelLagPackInput,
  expectPanelLagPackFit,
  createPanelLagMomentInput,
  createPanelLagMomentProbe,
  expectPanelLagMoments,
  getPanelLagMomentRecord
} from './panel-lag-pack.test-utils';
import {computeSpatialPanelLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';

it('tiled moments retain small within squares against the integer oracle', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    const input = createPanelLagMomentInput('unit');
    const record = getPanelLagMomentRecord(graph, 'unit', true);
    for (const binding of record.bindings) {
      if (binding.name === 'validity') memory.write(binding.view, new Float32Array(4));
      else if (binding.name !== 'partial') {
        const column = binding.name === 'predictors' ? 1 : binding.name === 'response' ? 2 : 3;
        memory.write(
          binding.view,
          Float32Array.from({length: 32}, (_, row) => input.data[row * 8 + column])
        );
      }
    }
    const node = createWGSLKernelNode(graph, {
      ...record,
      id: 'tiled-moment-probe',
      workgroupSize: record.workgroupSize[0]
    });
    executeCommandNodesOnCPU([node], memory);
    const values = memory.readFloat32(
      record.bindings.find(binding => binding.name === 'partial')!.view
    );
    expectPanelLagMoments(values.subarray(0, 3), 'unit', true);
  } finally {
    device.destroy();
  }
});

for (const effects of ['unit', 'random'] as const)
  it(`retains small within squares in the exact integer moment oracle: ${effects}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      const input = createPanelLagMomentInput(effects);
      const data = createTransientView(graph, 'moment-data', 'float32', input.data.length);
      const output = createTransientView(
        graph,
        'moment-output',
        'float32',
        effects === 'unit' ? 9 : 24
      );
      memory.write(data, input.data);
      executeCommandNodesOnCPU(
        [
          createPanelLagMomentProbe(
            graph,
            getPanelLagMomentRecord(graph, effects),
            output,
            effects,
            data
          )
        ],
        memory
      );
      expectPanelLagMoments(memory.readFloat32(output), effects);
    } finally {
      device.destroy();
    }
  });

for (const scale of [1, 2])
  it(`CPU unit fit matches the original-input SVD oracle with large effects, scale=${scale}`, () => {
    expectPanelLagPackFit(
      computeSpatialPanelLagMaximumLikelihoodOnCPU(createPanelLagPackInput(scale), {
        periodCount: 2,
        predictorCount: 1,
        effects: 'unit'
      }),
      scale
    );
  });

for (const effects of ['unit', 'random'] as const)
  for (const scale of [1, 2])
    it(`packs W times small within responses before adding large means: ${effects}, scale=${scale}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device),
          memory = createCPUGraphMemory();
        recordWGSLKernelNodes(graph);
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `pack-${serial++}`, format, length);
        const input = createPanelLagPackInput(scale);
        const props = {
          effects,
          periodCount: 2,
          predictorCount: 1,
          predictors: view('float32', 8),
          response: view('float32', 8),
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          output: {
            coefficients: view('float32', 4),
            covariance: view('float32', 9),
            summary: view('float32', 6),
            status: view('uint32', 1),
            fitted: view('float32', 8),
            residuals: view('float32', 8)
          }
        };
        memory.write(props.predictors, input.predictors);
        memory.write(props.response, input.response);
        memory.write(props.weights.offsets, input.weights.offsets);
        memory.write(props.weights.neighbors, input.weights.neighbors);
        memory.write(props.weights.weights, input.weights.weights);
        const nodes = new GPUSpatialPanelLagMaximumLikelihood(props)
          .getCommandNodes(graph)
          .filter(
            node =>
              node.id.includes('-within-pack-') ||
              ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
          );
        const pack = getWGSLKernelNodeRecord(
          graph,
          nodes.find(node => node.id.endsWith('-pack'))!.id
        )!;
        const data = pack.bindings.find(binding => binding.name === 'data')!.view;
        // Only emulate the absent ldexp builtin; every other pack operation is the actual WGSL.
        executeCommandNodesOnCPU(
          nodes.map(node => {
            const record = getWGSLKernelNodeRecord(graph, node.id)!;
            return createWGSLKernelNode(graph, {
              id: `${node.id}-interpreter`,
              operation: record.operation,
              invocationCount: record.invocationCount,
              workgroupSize: record.workgroupSize[0],
              bindings: [...record.bindings],
              body: record.body,
              declarations: `fn auditLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'auditLdexp(')}`
            });
          }),
          memory
        );
        const values = memory.readFloat32(data);
        for (let row = 0; row < 8; row++) {
          expect(Number.isFinite(values[row * 8 + 3])).toBe(true);
          expect(values[row * 8 + 2], `within response ${row}`).toBe(input.withinResponse[row]);
          expect(values[row * 8 + 3], `within lag ${row}`).toBe(input.withinLag[row]);
        }
      } finally {
        device.destroy();
      }
    });
