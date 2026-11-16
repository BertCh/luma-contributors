// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createHierarchicalDensityCoreNode} from '../../../src/gpu-spatial-analysis/spatial-clustering/hdbscan-kernels';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('times the isolated exact core stage on increasing coincident populations and two duplicate groups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const rows of [256, 4096, 65536, 1_000_000]) {
      for (const groups of [1, 2]) {
        const positions = new Float32Array(rows * 2);
        if (groups === 2) for (let row = rows / 2; row < rows; row++) positions[row * 2] = 10;
        const leafCapacity = 2 ** Math.ceil(Math.log2(rows));
        const internalNodeCount = leafCapacity - 1;
        const nodeCount = leafCapacity * 2 - 1;
        // CPU-built boxes isolate the actual production core kernel from BVH construction and
        // reconstruction costs. Sorted duplicate rows have an explicit, independent zero oracle.
        const nodeData = new Uint32Array(nodeCount * 4);
        const boxes = new Float32Array(nodeData.buffer);
        for (let slot = 0; slot < leafCapacity; slot++) {
          const base = (internalNodeCount + slot) * 4;
          boxes[base] = slot < rows ? positions[slot * 2] : 0;
          nodeData[base + 2] = slot < rows ? slot : 0xffffffff;
        }
        const minimumX = (node: number) =>
          node < internalNodeCount
            ? boxes[node * 4]
            : nodeData[node * 4 + 2] === 0xffffffff
              ? Infinity
              : boxes[node * 4];
        const maximumX = (node: number) =>
          node < internalNodeCount
            ? boxes[node * 4 + 2]
            : nodeData[node * 4 + 2] === 0xffffffff
              ? -Infinity
              : boxes[node * 4];
        for (let node = internalNodeCount - 1; node >= 0; node--) {
          boxes[node * 4] = Math.min(minimumX(node * 2 + 1), minimumX(node * 2 + 2));
          boxes[node * 4 + 2] = Math.max(maximumX(node * 2 + 1), maximumX(node * 2 + 2));
        }
        const graph = new GPUCommandGraph(device);
        const positionsBuffer = createInputBuffer(device, positions);
        const nodeDataBuffer = createInputBuffer(device, nodeData);
        const coreBuffer = createOutputBuffer(device, rows);
        graph.add(
          createHierarchicalDensityCoreNode(graph, {
            id: 'coincident-core',
            shape: {
              rowCount: rows,
              minimumSamples: 32,
              minimumClusterSize: 2,
              clusterCapacity: rows + 1,
              leafCapacity,
              internalNodeCount,
              levelCount: Math.log2(leafCapacity) + 1
            },
            positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
            nodeData: importGraphBuffer(graph, 'nodes', nodeDataBuffer, 'uint32', nodeData.length),
            coreDistances: importGraphBuffer(graph, 'cores', coreBuffer, 'float32', rows)
          })
        );
        const compiled = await graph.compileAsync();
        try {
          const started = performance.now();
          submitGraph(device, compiled, undefined);
          const actual = await readFloat32(coreBuffer, rows);
          const milliseconds = performance.now() - started;
          console.info(
            `hdbscan core rows=${rows} duplicateGroups=${groups} dispatchAndReadMilliseconds=${milliseconds.toFixed(2)}`
          );
          expect(actual.every(distance => distance === 0)).toBe(true);
          expect(milliseconds).toBeLessThan(60000);
        } finally {
          compiled.destroy();
          positionsBuffer.destroy();
          nodeDataBuffer.destroy();
          coreBuffer.destroy();
        }
      }
    }
  } finally {
    device.destroy();
  }
}, 900000);
