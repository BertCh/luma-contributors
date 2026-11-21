// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const operation of ['p-median', 'maximum-coverage'] as const) {
  for (const sparse of [false, true]) {
    it(`pins ${operation} ${sparse ? 'sparse' : 'dense'} swap source and node budget across extraction`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        let sequence = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `view-${sequence++}`, format, length);
        const model = new GPULocationAllocation({
          id: 'split',
          operation,
          facilityCount: 3,
          maximumFacilityCount: 2,
          maximumSwapIterations: 2,
          ...(sparse
            ? {
                sparseCosts: {
                  offsets: view('uint32', 4),
                  facilities: view('uint32', 6),
                  costs: view('float32', 6)
                }
              }
            : {costs: view('float32', 9)}),
          ...(operation === 'maximum-coverage' ? {coverageDistance: view('float32', 1)} : {}),
          output: {
            selectedFacilities: view('uint32', 3),
            assignments: view('uint32', 3),
            objective: view('float32', 1),
            status: view('uint32', 7)
          }
        });
        const nodes = model.getCommandNodes(graph);
        const kernels = getGPUCommandNodeKernelSources(
          nodes.filter(node => node.id.includes('-swap-') && !node.id.includes('-list'))
        );
        const digest = createHash('sha256').update(JSON.stringify(kernels)).digest('hex');
        // Captured before parallel compaction: pins all other swap WGSL bytes and node IDs.
        const expected =
          operation === 'p-median'
            ? sparse
              ? 'b320417f91a6a4b0b43ab0765c0b832a65e40e3bd48dc4bf92067e2456bbda0a'
              : 'cbbb44156f75f3f7e010fb23e29dd35f63e2f9879409fbf11a2a577e31817fa9'
            : sparse
              ? '504da658b5e599e9c45e8eadd2bd505b6258789745505f7253acf2ee1edee85f'
              : '177d5cf2d55a40d8dfb10c8dd1c8547ad7541225c6e2eb3e5b39dc1c270b2a64';
        const allKernels = getGPUCommandNodeKernelSources(nodes);
        expect(
          allKernels.some(
            ({id, source}) => id.includes('-list') && source.includes('facility < FACILITY_COUNT')
          )
        ).toBe(false);
        expect(allKernels.some(({id}) => id.endsWith('-list-scan-0'))).toBe(true);
        expect(digest).toBe(expected);
        expect(kernels.length).toBe(11);
        expect(nodes.length).toBeLessThanOrEqual(model.estimatedCommandNodeCount);
      } finally {
        device.destroy();
      }
    });
  }
}
