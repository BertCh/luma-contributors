// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createFacilityScoreReductionNodes} from '../../../src/gpu-spatial-analysis/location-allocation/facility-score-reduction';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

it('reduces lexicographic scores and invalid flags exactly across multiple levels and an unaligned first view', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const candidates = 4097;
    const values = new Uint32Array(16 + candidates * 4).fill(0xffffffff);
    for (let candidate = 0; candidate < candidates; candidate++)
      values.set([candidate % 3, 7, candidate, 1 << (candidate % 8)], 16 + candidate * 4);
    values.set([0, 6, 4000, 0], 16 + 4000 * 4);
    const storage = rig.input(values, 'uint32');
    const scores = rig.graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: candidates * 4,
      byteOffset: 64
    });
    const result = rig.output('uint32', 4);
    const reduction = createFacilityScoreReductionNodes(rig.graph, {
      id: 'facility-reduce',
      operation: 'FacilityScoreTest',
      scores,
      candidateCount: candidates
    });
    // Read through a caller output so the rig owns and releases its readback buffer.
    const {createWGSLKernelNode} = await import(
      '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
    );
    rig.run({
      getCommandNodes: () => [
        ...reduction.nodes,
        createWGSLKernelNode(rig.graph, {
          id: 'publish',
          operation: 'FacilityScoreTest',
          invocationCount: 4,
          bindings: [
            {name: 'winner', view: reduction.winner, type: 'u32', access: 'read'},
            {name: 'result', view: result, type: 'u32', access: 'read_write'}
          ],
          body: 'result[resultOffset + index] = winner[winnerOffset + index];'
        })
      ]
    });
    expect(await rig.readUint(result)).toEqual([0, 6, 4000, 255]);
  } finally {
    rig.destroy();
  }
});
