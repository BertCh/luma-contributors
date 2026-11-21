// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUStreamlines} from '../../../src/raster-analysis/flow-texture/gpu-streamlines';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expectPublicationOrder} from './flow-output-assertions';

it('streamline counts reduce simultaneous undecided seeds atomically after clearing state', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const word = (id: string, length = 1) => createTransientView(graph, id, 'uint32', length);
  const nodes = new GPUStreamlines({
    id: 'publication',
    velocities: createTransientView(graph, 'velocities', 'float32x2', 1),
    fieldWidth: 1,
    fieldHeight: 1,
    gridWidth: 1,
    gridHeight: 1,
    seedColumns: 4,
    seedRows: 1,
    stepsPerDirection: 1,
    maximumIterations: 1,
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    uint32Parameters: word('words', 4),
    output: {
      lines: {
        ids: word('ids', 4),
        status: {
          count: word('count'),
          overflow: word('overflow'),
          candidateOverflow: word('candidateOverflow')
        }
      },
      lineOffsets: word('offsets', 5),
      positions: createTransientView(graph, 'positions', 'float32x2', 12),
      positionCount: word('positionCount')
    }
  }).getCommandNodes(graph);
  const counts = getWGSLKernelNodeRecord(graph, 'publication-counts')!;
  expect(counts.invocationCount).toBe(4);
  expect(counts.bindings.find(binding => binding.name === 'state')?.type).toBe('atomic<u32>');
  expect(counts.body).toContain('atomicOr(&state[stateOffset + 1u], 1u);');
  expectPublicationOrder(nodes);
});
