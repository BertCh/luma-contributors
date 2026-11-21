// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {createPreparedGeometrySnapshot} from '../../../src/gpu-spatial-analysis/spatial-join/prepared-geometry-snapshot';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

type RecordedCopy = Parameters<CommandEncoder['copyBufferToBuffer']>[0];
type RecordedAction = {copy: RecordedCopy} | {node: GPUCommandNode<void>};

for (const generations of [
  [
    [1, 0],
    [2, 0]
  ],
  [
    [0x12345678, 0x9abcdef0],
    [0xfedcba98, 0x76543210]
  ],
  [
    [0xffffffff, 0x12345678],
    [0, 0x12345679]
  ]
]) {
  it(`ERG-02 records bounded immutable generation bytes and reverses geometry submissions ${generations[0]}`, async () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph<void>(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const ownedBuffers: Buffer[] = [];
    const generation = Uint32Array.from(generations[0]);
    const snapshot = createPreparedGeometrySnapshot(graph, {
      id: 'reverse-host',
      views: {positions},
      generation,
      ownedBuffers,
      recordEncoding: () => {}
    });
    const nodes = [...snapshot.nodes, snapshot.commit(snapshot.nodes)];
    const views = new Map<Buffer, GraphDataView>();
    for (const node of nodes) {
      for (const resource of node.resources ?? []) {
        if ('buffer' in resource && 'format' in resource.buffer) {
          const view = resource.buffer as GraphDataView;
          const defaultBuffer = view.buffer.defaultBuffer;
          if (defaultBuffer instanceof Buffer) views.set(defaultBuffer, view);
        }
      }
    }
    const allocation = vi.spyOn(device, 'createBuffer');
    const control = ownedBuffers.find(buffer => buffer.id === 'reverse-host-requested-generation')!;
    const write = vi.spyOn(control, 'write');
    const encode = (): RecordedAction[] => {
      const recorded: RecordedAction[] = [];
      for (const node of nodes) {
        if (node.condition?.source === 'cpu') expect(node.condition.evaluate(undefined)).toBe(true);
        if (node.type === 'copy') {
          const executable = node.compile({device});
          try {
            executable.encode({
              commandEncoder: {
                copyBufferToBuffer: (copy: RecordedCopy) => recorded.push({copy})
              } as unknown as CommandEncoder,
              parameters: undefined,
              getBuffer: value => {
                const defaultBuffer = ('buffer' in value ? value.buffer : value).defaultBuffer;
                if (!defaultBuffer) throw new Error('expected an imported buffer');
                return getCoreDefaultBuffer(defaultBuffer);
              },
              getTexture: () => {
                throw new Error('unexpected texture');
              },
              getTextureView: () => {
                throw new Error('unexpected texture view');
              },
              getExternalTexture: () => {
                throw new Error('unexpected external texture');
              }
            });
          } finally {
            executable.destroy?.();
          }
        } else {
          // CPU conditions were evaluated at recording, just as in the public graph encoder.
          recorded.push({
            node: node.condition?.source === 'cpu' ? {...node, condition: undefined} : node
          });
        }
      }
      return recorded;
    };
    try {
      const first = encode();
      generation.set(generations[1]);
      const second = encode();
      expect(allocation).not.toHaveBeenCalled();
      const memory = createCPUGraphMemory();
      // Initialize imported allocations after both encodings. This models the old shared queue
      // write faithfully: both command streams would see its latest value at submission time.
      for (const [buffer, view] of views) memory.write(view, await buffer.readAsync());
      const execute = (actions: RecordedAction[], geometry: Float32Array) => {
        memory.write(positions, geometry);
        for (const action of actions) {
          if ('copy' in action) {
            const copy = action.copy;
            const source = memory.getBuffer(views.get(copy.sourceBuffer)!.buffer);
            const destination = memory.getBuffer(views.get(copy.destinationBuffer)!.buffer);
            new Uint8Array(destination.backing, copy.destinationOffset ?? 0, copy.size).set(
              new Uint8Array(source.backing, copy.sourceOffset ?? 0, copy.size).slice()
            );
          } else {
            executeCommandNodesOnCPU([action.node], memory);
          }
        }
        return Array.from(memory.readFloat32(snapshot.views.positions));
      };
      // Literal, independent geometry: each stream must capture the point supplied immediately
      // before its own snapshot executes, regardless of the order in which it was encoded.
      expect(execute(second, Float32Array.of(207, 5))).toEqual([207, 5]);
      expect(execute(first, Float32Array.of(101, 11))).toEqual([101, 11]);
      const state = [...views.values()].find(
        view => view.buffer.id === 'reverse-host-applied-generation'
      )!;
      expect(Array.from(memory.readUint32(state, 2))).toEqual(generations[0]);
      expect(write).not.toHaveBeenCalled();
      for (const actions of [first, second]) {
        const copies = actions.filter(action => 'copy' in action);
        expect(copies).toHaveLength(8);
        for (const action of copies) {
          if ('copy' in action) {
            expect(action.copy.size).toBe(4);
            expect((action.copy.sourceOffset ?? 0) % 4).toBe(0);
            expect((action.copy.destinationOffset ?? 0) % 4).toBe(0);
          }
        }
      }
    } finally {
      allocation.mockRestore();
      write.mockRestore();
      for (const buffer of ownedBuffers) buffer.destroy();
      device.destroy();
    }
  });
}
