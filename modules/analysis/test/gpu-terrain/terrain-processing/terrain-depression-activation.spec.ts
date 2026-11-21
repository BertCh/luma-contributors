// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createTerrainDepressionActivationNodes} from '../../../src/gpu-terrain/terrain-processing/terrain-depression-activation';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('bounds work per lane while activating wide two-row plateaus exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const width of [1024, 2048]) {
    const count = width * 2;
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createOutputBuffer>[] = [];
    let identifier = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `plateau-input-${identifier++}`,
        buffer,
        format,
        values.length
      );
    };
    const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `plateau-output-${identifier++}`, buffer, format, length)
      };
    };
    const tree = output('uint32', count * 4);
    const levels = output('float32', count * 2);
    const ownership = output('uint32', count);
    const storedCount = output('uint32', 1);
    const requiredCount = output('uint32', 1);
    const overflow = output('uint32', 1);
    const maximumWork = output('uint32', 1);
    const elevation = input('float32', new Float32Array(count));
    const outlets = input('uint32', new Uint32Array(count));
    // A flat surface sorts by cell ID. Supplying this known order isolates activation from sort.
    const sortedCells = input(
      'uint32',
      Uint32Array.from({length: count}, (_, cell) => cell)
    );
    const nodes = createTerrainDepressionActivationNodes(graph, {
      id: 'plateau',
      elevation,
      sortedCells,
      tree: tree.view,
      levels: levels.view,
      spliceMaximumWork: maximumWork.view,
      hierarchy: {
        grid: {width, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        elevation,
        outlets,
        stages: createTransientView(graph, 'unused-stages', 'float32', 1),
        nodeCapacity: 1,
        output: {
          nodes: createTransientView(graph, 'unused-nodes', 'uint32', 4),
          levels: createTransientView(graph, 'unused-levels', 'float32', 2),
          cellNodes: ownership.view,
          status: {
            count: storedCount.view,
            requiredCount: requiredCount.view,
            overflow: overflow.view
          },
          curves: createTransientView(graph, 'unused-curves', 'float32', 2)
        }
      }
    });
    graph.add(nodes);
    const compiled = graph.compile();
    try {
      // Run the same compiled graph twice: resumed endpoint records and work counters must reset.
      for (let repetition = 0; repetition < 2; repetition++) {
        submitGraph(device, compiled, undefined);
        // Hand-derived oracle: every cell joins the component born at cell zero, with no merge.
        expect(Array.from(await readUint32(storedCount.buffer, 1))).toEqual([1]);
        expect(Array.from(await readUint32(requiredCount.buffer, 1))).toEqual([1]);
        expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([0]);
        expect(Array.from(await readUint32(tree.buffer, 4))).toEqual([
          0xffffffff, 0, 0xffffffff, 1
        ]);
        expect(Array.from(await readFloat32(levels.buffer, 2))).toEqual([0, Infinity]);
        expect(Array.from(await readUint32(ownership.buffer, count))).toEqual(Array(count).fill(0));
        const maximumAttempts = (await readUint32(maximumWork.buffer, 1))[0];
        expect(maximumAttempts).toBeGreaterThan(0);
        // This is the measured maximum over all lanes and all rounds, not a host estimate.
        expect(maximumAttempts).toBeLessThanOrEqual(4096);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
}, 120_000);
