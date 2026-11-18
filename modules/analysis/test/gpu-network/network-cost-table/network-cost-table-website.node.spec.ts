// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {computeNetworkCostTableOnCPU} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import type {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

vi.mock('../../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));
const SCENE_MODULE =
  '../../../../../website/src/gpu/runs/entries/network-routing-access/network-cost-table';
type SceneArray = Uint32Array | Float32Array;
type SceneBuffer = {id: string; view: GraphDataView; write(values: SceneArray): void};
type SceneInputs = {
  node: number;
  network: {
    nodeCount: number;
    nodes: Float32Array;
    offsets: Uint32Array;
    neighbors: Uint32Array;
    lengths: Float32Array;
  };
};
type SceneBuilder = {
  graph: GPUCommandGraph;
  upload(
    id: string,
    format: 'uint32' | 'float32' | 'uint32x2' | 'float32x2',
    values: SceneArray
  ): SceneBuffer;
  allocate(id: string, format: 'uint32' | 'float32', length: number): SceneBuffer;
};
type SceneSpecification = {
  createInputs(context: {random(): number; world: {width: number; height: number}}): SceneInputs;
  build(
    builder: SceneBuilder,
    inputs: SceneInputs
  ): {
    outputs: Record<string, SceneBuffer>;
    update(values: {costLimit: number; congestion: number}): void;
  };
  interaction: {
    resolve(
      pointer: {x: number; y: number},
      context: {
        inputs: SceneInputs;
        outputs: Record<string, SceneBuffer>;
        values: {costLimit: number; congestion: number};
        peek(buffer: SceneBuffer): SceneArray | null;
      }
    ): {readout: string};
  };
};

it('marks the truncated node-143 demo row incomplete and reports 32 targets at full capacity', async () => {
  const scene = (await import(SCENE_MODULE)).default as SceneSpecification;
  const inputs = scene.createInputs({random: () => 0.5, world: {width: 1, height: 1}});
  inputs.node = 143;
  const values = {costLimit: 0.4, congestion: 0};
  const device = createNullWebGPUDevice();
  try {
    for (const capacity of [3000, 7834]) {
      const graph = new GPUCommandGraph(device);
      const add = vi.spyOn(graph, 'add');
      const arrays = new Map<SceneBuffer, SceneArray>();
      const createBuffer = (
        id: string,
        format: 'uint32' | 'float32' | 'uint32x2' | 'float32x2',
        values: SceneArray
      ): SceneBuffer => {
        const buffer = {
          id,
          view: createTransientView(
            graph,
            id,
            format,
            values.length / (format.endsWith('x2') ? 2 : 1)
          ),
          write: (updated: SceneArray) => values.set(updated)
        };
        arrays.set(buffer, values);
        return buffer;
      };
      const builder: SceneBuilder = {
        graph,
        upload: createBuffer,
        allocate: (id, format, length) =>
          createBuffer(
            id,
            format,
            format === 'uint32'
              ? new Uint32Array(id === 'targets' ? capacity : length)
              : new Float32Array(id === 'costs' ? capacity : length)
          )
      };
      const built = scene.build(builder, inputs);
      built.update(values);
      const result = computeNetworkCostTableOnCPU(
        {
          topology: {
            offsets: inputs.network.offsets,
            neighbors: inputs.network.neighbors,
            weights: inputs.network.lengths
          },
          costLimit: Float32Array.of(0.4)
        },
        {capacity, maximumIterations: 48}
      );
      expect(result.requiredCount[0]).toBe(7834);
      expect(result.offsets[144] - result.offsets[143]).toBe(capacity === 3000 ? 0 : 32);
      for (const name of ['offsets', 'targets', 'costs', 'complete'] as const) {
        const buffer = built.outputs[name];
        if (buffer) arrays.get(buffer)!.set(result[name]);
      }
      const readout = scene.interaction.resolve(
        {x: inputs.network.nodes[286], y: inputs.network.nodes[287]},
        {inputs, outputs: built.outputs, values, peek: buffer => arrays.get(buffer) ?? null}
      ).readout;
      expect(readout).toBe(
        capacity === 3000 ? 'node 143 · incomplete table' : 'node 143 · 32 targets within 0.40'
      );
      const contributor = add.mock.calls[0][0] as GPUNetworkCostTable;
      expect(contributor.props.output.status?.complete).toBe(built.outputs['complete'].view);
      if (capacity === 3000)
        expect(arrays.get(built.outputs['rowCosts'])!.every(Number.isNaN)).toBe(true);
      add.mockRestore();
    }
  } finally {
    device.destroy();
  }
});
