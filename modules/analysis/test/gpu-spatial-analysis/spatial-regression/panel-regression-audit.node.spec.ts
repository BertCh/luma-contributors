// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createPanelAuditInput} from './panel-audit.test-utils';

it('R7 unit IV removes large entity offsets without rounding away the within response', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `r7-${serial++}`, format, length);
    const input = createPanelAuditInput();
    input.response = Float32Array.from(
      input.response,
      (value, row) => value / 4096 + (row % 2 ? -3000 : 3000)
    );
    const props = {
      id: 'r7',
      effects: 'unit' as const,
      periodCount: 4,
      predictorCount: 1,
      predictors: view('float32', 16),
      response: view('float32', 16),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 16),
        status: {code: view('uint32', 1)}
      }
    };
    memory.write(props.predictors, input.predictors);
    memory.write(props.response, input.response);
    memory.write(props.weights.offsets, input.weights.offsets);
    memory.write(props.weights.neighbors, input.weights.neighbors);
    memory.write(props.weights.weights, input.weights.weights);
    const nodes = new GPUSpatialPanelRegression(props).getCommandNodes(graph);
    const firstFit = nodes.findIndex(node => node.id === 'r7-pooled-lag-weights-norm-tiles');
    executeCommandNodesOnCPU(
      nodes.slice(0, firstFit).map(node => {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        return createWGSLKernelNode(graph, {
          ...record,
          id: `${node.id}-interpreter`,
          workgroupSize: record.workgroupSize[0],
          declarations: `fn reviewLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'reviewLdexp(')}`
        });
      }),
      memory
    );
    const response = getWGSLKernelNodeRecord(graph, nodes[firstFit].id)!.bindings.find(
      binding => binding.name === 'response'
    )!.view;
    const original = createPanelAuditInput();
    // Each original entity mean is an exact quarter-integer; the original-input transform
    // is therefore exactly (y - [3.75,1.25,2.5,3.5]) / 4096, despite the +/-3000 shifts.
    expect(Array.from(memory.readFloat32(response))).toEqual(
      Array.from(
        original.response,
        (value, row) => (value - [3.75, 1.25, 2.5, 3.5][row % 4]) / 4096
      )
    );
  } finally {
    device.destroy();
  }
});

it('panel retains validated period/effect settings and output maps after caller mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `snapshot-audit-${serial++}`, format, length);
    const props = {
      periodCount: 4,
      predictorCount: 1,
      effects: 'pooled' as 'pooled' | 'unit',
      predictors: view('float32', 16),
      response: view('float32', 16),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 16),
        status: {code: view('uint32', 1)}
      }
    };
    const contributor = new GPUSpatialPanelRegression(props);
    const originalStatus = props.output.status.code;
    props.periodCount = 2;
    props.effects = 'unit';
    props.output.status.code = view('uint32', 1);
    expect(contributor.props.periodCount).toBe(4);
    expect(contributor.props.effects).toBe('pooled');
    expect(contributor.props.output.status.code).toBe(originalStatus);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const effects of ['pooled', 'unit'] as const)
  for (const malformed of ['terminal', 'descending', 'first', 'neighbor'] as const)
    it(`panel ${effects} refuses malformed ${malformed} CSR instead of fitting clamped edges`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        recordWGSLKernelNodes(graph);
        const memory = createCPUGraphMemory();
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `audit-${serial++}`, format, length);
        const data = createPanelAuditInput();
        if (malformed === 'terminal') data.weights.offsets[4] = 9;
        else if (malformed === 'descending') data.weights.offsets[2] = 1;
        else if (malformed === 'neighbor') data.weights.neighbors[0] = 4;
        else {
          data.weights.offsets = Uint32Array.from(data.weights.offsets, value => value + 1);
          data.weights.neighbors = Uint32Array.from([0, ...data.weights.neighbors]);
          data.weights.weights = Float32Array.from([0, ...data.weights.weights]);
        }
        const props = {
          id: 'audit-panel-regression',
          effects,
          periodCount: 4,
          predictorCount: 1,
          predictors: view('float32', 16),
          response: view('float32', 16),
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', data.weights.neighbors.length),
            weights: view('float32', data.weights.weights.length)
          },
          output: {
            table: view('float32', 12),
            summary: view('float32', 16),
            status: {code: view('uint32', 1)}
          }
        };
        memory.write(props.predictors, data.predictors);
        memory.write(props.response, data.response);
        memory.write(props.weights.offsets, data.weights.offsets);
        memory.write(props.weights.neighbors, data.weights.neighbors);
        memory.write(props.weights.weights, data.weights.weights);
        const nodes = new GPUSpatialPanelRegression(props).getCommandNodes(graph);
        const firstFit = nodes.findIndex(
          node => node.id === 'audit-panel-regression-pooled-lag-weights-norm-tiles'
        );
        expect(firstFit).toBeGreaterThan(0);
        executeCommandNodesOnCPU(
          nodes.slice(0, firstFit).map(node => {
            const record = getWGSLKernelNodeRecord(graph, node.id)!;
            return createWGSLKernelNode(graph, {
              ...record,
              id: `${node.id}-interpreter`,
              workgroupSize: record.workgroupSize[0],
              declarations: `fn reviewLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'reviewLdexp(')}`
            });
          }),
          memory
        );
        const response = getWGSLKernelNodeRecord(graph, nodes[firstFit].id)!.bindings.find(
          binding => binding.name === 'response'
        )!.view;
        // An out-of-capacity terminal offset or a backwards row has no well-defined W.
        // Poisoning observations forces the existing solver to withhold a numerical fit.
        expect(Array.from(memory.readFloat32(response)).some(Number.isNaN)).toBe(true);
      } finally {
        device.destroy();
      }
    });
