// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {computeNearestFeatureJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {createKernelAuditContext} from './kernel-audit-context';
import {CELLS_AND_JOINS_CASES} from './cases-cells-and-joins';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {auditKernelCase} from './kernel-audit-runner';

it('NAM-03 rejects the former lowest-id tie spelling', () => {
  const sources = (ties: 'lowest-row' | 'lowest-id') => {
    const context = createKernelAuditContext('row-tie-alias');
    const {view, graph} = context;
    try {
      const contributor = new GPUNearestFeatureJoin({
        id: 'tie-alias',
        queries: {kind: 'points', positions: view('float32x2', 1)},
        features: {kind: 'points', positions: view('float32x2', 2), sourceIds: view('uint32', 2)},
        neighborCount: 1,
        neighborCapacity: 1,
        ties: ties as 'lowest-row',
        output: {
          neighborRows: view('uint32', 1),
          neighborIds: view('uint32', 1),
          neighborCounts: view('uint32', 1),
          status: {overflow: view('uint32', 1)}
        }
      });
      return getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).map(
        module => module.source
      );
    } finally {
      context.device.destroy();
    }
  };
  expect(() => sources('lowest-row')).not.toThrow();
  expect(() => sources('lowest-id')).toThrow("ties 'lowest-id' was renamed; use 'lowest-row'");
  const actual = computeNearestFeatureJoinOnCPU(
    {kind: 'points', positions: Float32Array.of(0, 0)},
    {kind: 'points', positions: Float32Array.of(-1, 0, 1, 0), sourceIds: Uint32Array.of(99, 1)}
  );
  // Both distances are exactly 1; row 0 wins although application ID 1 is on row 1.
  expect(Array.from(actual.neighborRows)).toEqual([0]);
  expect(Array.from(actual.neighborIds)).toEqual([99]);
  expect(Number.isFinite(actual.neighborDistances[0])).toBe(true);
  expect(Array.from(actual.neighborDistances)).toEqual([1]);
});

const neighborGatherCases = CELLS_AND_JOINS_CASES.filter(
  auditCase =>
    /^(inner|left) neighbors every output$/.test(auditCase.name) &&
    auditCase.exportName === 'GPUPairGather'
);
it('NAM-03 retains both optional-status-absent neighbor gather cases', () => {
  expect(neighborGatherCases).toHaveLength(2);
});
for (const auditCase of neighborGatherCases) {
  it(`NAM-03 preserves optional-status-absent ${auditCase.name}`, () => {
    expect(auditKernelCase(auditCase).findings).toEqual([]);
  });
}
