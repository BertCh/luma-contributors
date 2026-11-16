// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createSpatialJoinFinalizeNode} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getReservedWGSLWords} from './wgsl-reserved-words';

// GPUCompactOutput documents aggregate completeness: an intermediate overflow makes the compact
// result incomplete even when the retained matches fit its final capacity. These expectations are
// explicit rather than inferred from the implementation or from packed/chunked agreement.
const STATUS_CASES = [
  {
    name: 'candidate capacity four',
    candidateCapacity: 4,
    matchCapacity: 20,
    state: [10, 0, 4, 0],
    leafOverflow: 0,
    expected: [1, 10, 0, 4, 1, 4]
  },
  {
    name: 'candidate capacity five',
    candidateCapacity: 5,
    matchCapacity: 20,
    state: [10, 0, 5, 0],
    leafOverflow: 0,
    expected: [1, 10, 0, 5, 1, 5]
  },
  {
    name: 'BVH leaf capacity only',
    candidateCapacity: 32,
    matchCapacity: 20,
    state: [2, 0, 2, 0],
    leafOverflow: 1,
    expected: [1, 2, 0, 2, 1, 2]
  },
  {
    name: 'compact capacity only',
    candidateCapacity: 32,
    matchCapacity: 4,
    state: [10, 0, 10, 0],
    leafOverflow: 0,
    expected: [1, 10, 0, 4, 1, 10]
  },
  {
    name: 'candidate and compact capacities',
    candidateCapacity: 4,
    matchCapacity: 3,
    state: [10, 0, 4, 0],
    leafOverflow: 0,
    expected: [1, 10, 0, 3, 1, 4]
  },
  {
    name: 'saturated candidate count',
    candidateCapacity: 4,
    matchCapacity: 20,
    state: [0xffffffff, 0, 4, 0],
    leafOverflow: 0,
    expected: [1, 0xffffffff, 0, 4, 1, 4]
  },
  {
    name: 'exact capacities with uncertainty',
    candidateCapacity: 7,
    matchCapacity: 4,
    state: [7, 3, 4, 0],
    leafOverflow: 0,
    expected: [0, 7, 3, 4, 0, 4]
  }
];

for (const testCase of STATUS_CASES) {
  it(`shared join finalizer executes aggregate statuses and clears subsequent frames: ${testCase.name}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const node = createSpatialJoinFinalizeNode(graph, {
      id: 'join-finalize-status',
      operation: 'JoinFinalizeStatus',
      state: createTransientView(graph, 'state', 'uint32', 4),
      bvhOverflow: scalar('bvh-overflow'),
      candidateCapacity: testCase.candidateCapacity,
      overflow: scalar('overflow'),
      candidateCount: scalar('candidate-count'),
      uncertainCount: scalar('uncertain-count'),
      matches: {
        ids: createTransientView(graph, 'ids', 'uint32', testCase.matchCapacity),
        status: {
          count: scalar('match-count'),
          overflow: scalar('match-overflow'),
          requiredCount: scalar('required-count')
        }
      }
    });
    try {
      const [kernel] = getGPUCommandNodeKernelSources([node]);
      expect(getReservedWGSLWords(kernel.source)).toEqual([]);
      const outputNames = [
        'overflow',
        'candidateCount',
        'uncertainCount',
        'matchCount',
        'matchOverflow',
        'matchTotalCount'
      ];
      const buffers: Record<string, Uint32Array> = {
        state: new Uint32Array(4),
        bvhOverflow: new Uint32Array(1),
        ...Object.fromEntries(outputNames.map(name => [name, new Uint32Array([0xfffffffe])]))
      };
      const bindings = Object.fromEntries(
        new WgslReflect(kernel.source).storage.map(declaration => [
          declaration.binding,
          buffers[declaration.name]
        ])
      );
      const execute = (state: number[], leafOverflow: number) => {
        buffers['state'].set(state);
        buffers['bvhOverflow'][0] = leafOverflow;
        // Execute the generated production shader without rewriting its body or constants.
        new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
          'main',
          [1, 1, 1],
          {0: bindings},
          {constants: kernel.constants}
        );
        return outputNames.map(name => buffers[name][0]);
      };
      expect(execute(testCase.state, testCase.leafOverflow)).toEqual(testCase.expected);
      // Keep output memory between frames: every status must be overwritten, including uncertainty.
      expect(execute([0, 0, 0, 0], 0)).toEqual([0, 0, 0, 0, 0, 0]);
      expect(execute([2, 0, 2, 0], 0)).toEqual([0, 2, 0, 2, 0, 2]);
    } finally {
      device.destroy();
    }
  });
}
