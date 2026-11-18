// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {COMMUTE_FLOWS} from '../../../../../website/src/gpu/data/commute-flows';
import {computeFlowAggregation} from './flow-aggregation-oracle';

vi.mock('../../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));
const SCENE_MODULE =
  '../../../../../website/src/gpu/runs/entries/network-routing-access/flow-aggregation';
type SceneArray = Float32Array | Uint32Array;
type SceneFormat = 'float32' | 'uint32' | 'float32x2';
type SceneBuffer = {
  id: string;
  format: SceneFormat;
  length: number;
  view: GraphDataView;
  buffer: Buffer;
  write(data: SceneArray): void;
};
type SceneValues = {gridSize: number; excludeSelfFlows: number; minimumJobs: number};
type SceneInputs = {
  world: {width: number; height: number};
  origins: Float32Array;
  destinations: Float32Array;
};
type SceneSpecification = {
  createInputs(context: {world: SceneInputs['world']}): SceneInputs;
  build(
    builder: {
      graph: GPUCommandGraph;
      upload(id: string, format: SceneFormat, data: SceneArray): SceneBuffer;
      allocate(id: string, format: SceneFormat, length: number): SceneBuffer;
    },
    inputs: SceneInputs,
    values: SceneValues
  ): {
    outputs: Record<string, SceneBuffer>;
    update(values: SceneValues): void;
  };
  interaction: {
    resolve(
      pointer: {x: number; y: number},
      context: {
        outputs: Record<string, SceneBuffer>;
        values: SceneValues;
        world: SceneInputs['world'];
        peek(buffer: SceneBuffer): SceneArray | null;
      }
    ): {readout: string; glyphs: {shape: string}[]};
  };
};

it('commute thresholds select accepted rows and clear rejected pairs on the same GPU graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = (await import(SCENE_MODULE)).default as SceneSpecification;
  const world = {width: 16 / 9, height: 1};
  const inputs = scene.createInputs({world});
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const wrap = (id: string, format: SceneFormat, buffer: Buffer, length: number): SceneBuffer => {
    buffers.push(buffer);
    return {
      id,
      format,
      length,
      buffer,
      view: importGraphBuffer(graph, id, buffer, format, length),
      write: data => buffer.write(data)
    };
  };
  const values = {gridSize: 1, excludeSelfFlows: 1, minimumJobs: 25};
  const built = scene.build(
    {
      graph,
      upload: (id, format, data) =>
        wrap(
          id,
          format,
          createInputBuffer(device, data),
          data.length / (format.endsWith('x2') ? 2 : 1)
        ),
      allocate: (id, format, length) =>
        wrap(
          id,
          format,
          createOutputBuffer(device, length * (format.endsWith('x2') ? 2 : 1)),
          length
        )
    },
    inputs,
    values
  );
  const compiled = graph.compile();
  try {
    for (const minimumJobs of [25, 400]) {
      const current = {...values, minimumJobs};
      built.update(current);
      submitGraph(device, compiled, undefined);
      const arrays = new Map<SceneBuffer, SceneArray>();
      for (const output of Object.values(built.outputs)) {
        const count = output.length * (output.format.endsWith('x2') ? 2 : 1);
        arrays.set(
          output,
          output.format.startsWith('float')
            ? Float32Array.from(await readFloat32(output.buffer, count))
            : Uint32Array.from(await readUint32(output.buffer, count))
        );
      }
      const count = arrays.get(built.outputs['count'])![0];
      expect(count).toBe(minimumJobs === 25 ? 160 : 16);
      expect(arrays.get(built.outputs['overflow'])![0]).toBe(Number(minimumJobs === 25));
      expect(arrays.get(built.outputs['pairOverflow'])![0]).toBe(0);
      const origins = arrays.get(built.outputs['originZones'])!;
      const destinations = arrays.get(built.outputs['destinationZones'])!;
      const weights = arrays.get(built.outputs['flowWeights'])!;
      const accepted = computeFlowAggregation({
        zones: {kind: 'grid', bounds: [0, 0, world.width, 1], gridSize: [12, 7]},
        origins: Array.from(inputs.origins),
        destinations: Array.from(inputs.destinations),
        excludeSelfFlows: true,
        weights: COMMUTE_FLOWS.map(([, , jobs]) => jobs),
        mask: COMMUTE_FLOWS.map(([, , jobs]) => Number(jobs >= minimumJobs))
      });
      expect(accepted.flows.length).toBe(minimumJobs === 25 ? 205 : 16);
      const acceptedPairs = new Set(accepted.flows.map(flow => flow.pairKey));
      expect(Array.from(weights.subarray(0, count)).every(weight => weight > 0)).toBe(true);
      const readout = scene.interaction.resolve(
        {x: (3.5 * world.width) / 12, y: 1 - 0.5 / 7},
        {
          outputs: built.outputs,
          values: current,
          world,
          peek: buffer => arrays.get(buffer) ?? null
        }
      );
      const outgoing = Array.from(origins.subarray(0, count)).filter(origin => origin === 3).length;
      expect(readout.glyphs.filter(glyph => glyph.shape === 'arrow')).toHaveLength(outgoing);
      expect(readout.readout).toBe(
        minimumJobs === 25
          ? 'zone 3 · out 714 jobs · 3 displayed flows · top 507'
          : 'zone 3 · out 0 jobs · 0 flows'
      );
      for (let flow = 0; flow < count; flow++) {
        expect(destinations[flow]).toBeLessThan(84);
        expect(acceptedPairs.has(origins[flow] * 84 + destinations[flow])).toBe(true);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}, 120000);
