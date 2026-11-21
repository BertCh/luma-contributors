// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectorySegments} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

it('publishes exact trip prefixes and the terminal total across zero-trip tracks and workgroups', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `prefix-${serial++}`, format, length);
  recordWGSLKernelNodes(graph);
  const contributor = new GPUTrajectorySegments({
    id: 'prefix',
    positions: view('float32x2', 12),
    timestamps: view('float32', 12),
    trackOffsets: view('uint32', 514),
    staypointIds: view('uint32', 12),
    parameters: view('float32', 4),
    staypoints: {
      tracks: view('uint32', 1),
      startRows: view('uint32', 1),
      finishRows: view('uint32', 1),
      durations: view('float32', 1),
      status: {count: view('uint32', 1)}
    },
    output: {
      triplegs: {
        ids: view('uint32', 12),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      trips: {
        ids: view('uint32', 2),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        }
      }
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  const terminal = getWGSLKernelNodeRecord(graph, 'prefix-trip-offsets')!;
  const binding = (name: string) => terminal.bindings.find(entry => entry.name === name)!.view;
  const memory = createCPUGraphMemory();
  const counts = new Uint32Array(513);
  counts[1] = 2;
  counts[257] = 1;
  counts[512] = 3;
  memory.write(binding('tripCounts'), counts);
  executeCommandNodesOnCPU(
    nodes.filter(
      node => node.id.startsWith('prefix-trip-offset-scan') || node.id === 'prefix-trip-offsets'
    ),
    memory
  );
  // Closed-form prefix: two trips at track 1, one at 257, three at 512; all other counts zero.
  expect(Array.from(memory.readUint32(binding('trackTripOffsets')))).toEqual([
    0,
    0,
    ...new Array<number>(256).fill(2),
    ...new Array<number>(255).fill(3),
    6
  ]);
  expect(Array.from(memory.readUint32(binding('tripTotal')))).toEqual([6]);
});
