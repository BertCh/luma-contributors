// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getTiledKeyUnionNodes} from '../../src/analysis-core/kernels/tiled-key-union';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../src/testing/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('unions offset uint64 keys including the maximum key, with saturated counts and bounded output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [2, 4]) {
    const buffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const graph = new GPUCommandGraph(device);
      const rows = [
        [1n, 3n],
        [1n, 2n, 0xffffffffffffffffn],
        [2n, 0xffffffffffffffffn]
      ];
      const counts = [
        [0xfffffffe, 1],
        [4, 5, 6],
        [7, 8]
      ];
      const oracle = new Map<bigint, number>();
      const importWords = (name: string, words: Uint32Array) => {
        const buffer = device.createBuffer({data: words, usage: Buffer.STORAGE});
        buffers.push(buffer);
        return importGraphBuffer(graph, name, buffer, 'uint32');
      };
      const tiles = rows.map((keys, tileIndex) => {
        const prefix = tileIndex + 1;
        const words = new Uint32Array(prefix + 8).fill(0xffffffff);
        const paddedCounts = new Uint32Array(4);
        for (const [row, key] of keys.entries()) {
          words[prefix + 2 * row] = Number(key & 0xffffffffn);
          words[prefix + 2 * row + 1] = Number(key >> 32n);
          paddedCounts[row] = counts[tileIndex][row];
          oracle.set(key, Math.min(0xffffffff, (oracle.get(key) ?? 0) + counts[tileIndex][row]));
        }
        const keysView = importWords(`keys-${tileIndex}`, words);
        return {
          keys: graph.createDataView(keysView.buffer, {
            format: 'uint32x2',
            length: 4,
            byteOffset: prefix * 4
          }),
          counts: importWords(`counts-${tileIndex}`, paddedCounts),
          count: importWords(`count-${tileIndex}`, Uint32Array.of(keys.length))
        };
      });
      const outputView = (name: string, length: number) => {
        const buffer = createOutputBuffer(device, length);
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
      };
      const keys = outputView('keys', 2 * capacity);
      const summedCounts = outputView('counts', capacity);
      const count = outputView('count', 1);
      const overflow = outputView('overflow', 1);
      const required = outputView('required', 1);
      const result = getTiledKeyUnionNodes(
        graph,
        'union',
        tiles,
        {
          keys: graph.createDataView(keys.view.buffer, {format: 'uint32x2', length: capacity}),
          counts: summedCounts.view
        },
        {count: count.view, overflow: overflow.view, requiredCount: required.view}
      );
      for (const node of result.nodes) graph.add(node);
      compiled = graph.compile();
      for (let repetition = 0; repetition < 2; repetition++) {
        submitGraph(device, compiled, undefined);
        const expected = [...oracle]
          .sort(([first], [second]) => (first < second ? -1 : first > second ? 1 : 0))
          .slice(0, capacity);
        expect(await readUint32(keys.buffer)).toEqual(
          expected.flatMap(([key]) => [Number(key & 0xffffffffn), Number(key >> 32n)])
        );
        expect(await readUint32(summedCounts.buffer)).toEqual(expected.map(([, total]) => total));
        expect(await readUint32(count.buffer)).toEqual([capacity]);
        expect(await readUint32(required.buffer)).toEqual([4]);
        expect(await readUint32(overflow.buffer)).toEqual([capacity < 4 ? 1 : 0]);
      }
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
