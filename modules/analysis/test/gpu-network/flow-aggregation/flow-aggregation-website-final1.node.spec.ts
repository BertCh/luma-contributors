// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import type {GPUFlowAggregation} from '../../../src/gpu-network/flow-aggregation/gpu-flow-aggregation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeFlowAggregation} from './flow-aggregation-oracle';
import {COMMUTE_FLOWS} from '../../../../../website/src/gpu/data/commute-flows';

vi.mock('../../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));
const SCENE_MODULE =
  '../../../../../website/src/gpu/runs/entries/network-routing-access/flow-aggregation';
type SceneArray = Float32Array | Uint32Array;
type SceneBuffer = {
  id: string;
  format: string;
  length: number;
  view: GraphDataView;
  write(values: SceneArray): void;
};
type SceneInputs = {
  world: {width: number; height: number};
  origins: Float32Array;
  destinations: Float32Array;
};
type SceneValues = {gridSize: number; excludeSelfFlows: number; minimumJobs: number};
type SceneSpecification = {
  createInputs(context: {world: SceneInputs['world']}): SceneInputs;
  build(
    builder: {
      graph: GPUCommandGraph;
      upload(id: string, format: string, values: SceneArray): SceneBuffer;
      allocate(id: string, format: string, length: number): SceneBuffer;
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

for (const minimumJobs of [25, 400]) {
  it(`commute flow readout uses accepted rows and reports displayed pairs at threshold ${minimumJobs}`, async () => {
    const scene = (await import(SCENE_MODULE)).default as SceneSpecification;
    const world = {width: 16 / 9, height: 1};
    const inputs = scene.createInputs({world});
    const values = {gridSize: 1, excludeSelfFlows: 1, minimumJobs};
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const add = vi.spyOn(graph, 'add');
    const arrays = new Map<GraphDataView, SceneArray>();
    const wrap = (id: string, format: string, data: SceneArray): SceneBuffer => {
      const view = createTransientView(
        graph,
        id,
        format as 'float32' | 'float32x2' | 'uint32' | 'uint32x2',
        data.length / (format.endsWith('x2') ? 2 : 1)
      );
      arrays.set(view, data);
      return {id, format, length: view.length, view, write: updated => data.set(updated)};
    };
    const built = scene.build(
      {
        graph,
        upload: wrap,
        allocate: (id, format, length) =>
          wrap(id, format, format === 'uint32' ? new Uint32Array(length) : new Float32Array(length))
      },
      inputs,
      values
    );
    built.update(values);
    for (const missing of ['all', 'count', 'overflow', 'pairOverflow', 'zoneOutWeights']) {
      const pending = scene.interaction.resolve(
        {x: (3.5 * world.width) / 12, y: 1 - 0.5 / 7},
        {
          outputs: built.outputs,
          values,
          world,
          peek: buffer =>
            missing === 'all' || buffer === built.outputs[missing]
              ? null
              : (arrays.get(buffer.view) ?? null)
        }
      );
      expect.soft(pending.readout).toBe('zone 3 · pending aggregation');
      expect(pending.glyphs.filter(glyph => glyph.shape === 'arrow')).toHaveLength(0);
    }
    const producer = add.mock.calls[0][0] as GPUFlowAggregation;
    const weightsView = producer.props.weights;
    if (!weightsView || !('buffer' in weightsView))
      throw new Error('Website weights must be a packed view');
    const result = computeFlowAggregation({
      zones: {kind: 'grid', bounds: [0, 0, world.width, 1], gridSize: [12, 7]},
      origins: Array.from(inputs.origins),
      destinations: Array.from(inputs.destinations),
      weights: Array.from(arrays.get(weightsView)!),
      mask: producer.props.mask ? Array.from(arrays.get(producer.props.mask)!) : undefined,
      excludeSelfFlows: true
    });
    const displayed = result.flows.slice(0, 160);
    expect(displayed.every(flow => flow.weight > 0)).toBe(true);
    expect(result.flows.length).toBe(minimumJobs === 25 ? 205 : 16);
    const output = producer.props.output;
    const write = (view: GraphDataView | undefined, data: number[]) => {
      if (view) arrays.get(view)!.set(data);
    };
    write(
      output.flowOriginZoneIds,
      displayed.map(flow => flow.originZone)
    );
    write(
      output.flowDestinationZoneIds,
      displayed.map(flow => flow.destinationZone)
    );
    write(
      output.flowWeights,
      displayed.map(flow => flow.weight)
    );
    write(output.zoneOutWeights, result.zoneOutWeights);
    write(output.status.count, [displayed.length]);
    write(output.status.overflow, [Number(result.flows.length > 160)]);
    write(output.status.pairOverflow, [0]);
    const readout = scene.interaction.resolve(
      {x: (3.5 * world.width) / 12, y: 1 - 0.5 / 7},
      {
        outputs: built.outputs,
        values,
        world,
        peek: buffer => (buffer ? (arrays.get(buffer.view) ?? null) : null)
      }
    );
    const outgoing = displayed.filter(flow => flow.originZone === 3);
    expect(readout.glyphs.filter(glyph => glyph.shape === 'arrow')).toHaveLength(outgoing.length);
    expect(readout.readout).toBe(
      minimumJobs === 25
        ? 'zone 3 · out 714 jobs · 3 displayed flows · top 507'
        : 'zone 3 · out 0 jobs · 0 flows'
    );
    expect(producer.props.mask).toBeDefined();
    expect(arrays.get(producer.props.mask!)).toEqual(
      Uint32Array.from(COMMUTE_FLOWS, ([, , jobs]) => Number(jobs >= minimumJobs))
    );
    device.destroy();
  });
}
