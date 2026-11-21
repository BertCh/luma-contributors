// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, writeFileSync} from 'node:fs';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createReplayTest} from './local-permutation-test-fixture';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {findStatisticsNonFiniteConstants} from '../spatial-weights/statistics-wgsl-audit';

// The fixture is `JSON.stringify(kernels, null, 2)` plus a newline, the full WGSL of the permute
// kernel per configuration. When a shared WGSL module that the kernel includes changes on purpose,
// regenerate it with `UPDATE_LOCAL_PERMUTATION_KERNELS=1 node_modules/.bin/vitest run --project node
// modules/analysis/test/gpu-spatial-analysis/computation-pack/local-permutation-kernels.node.spec.ts`
// --maxWorkers=1 and diff the old and new text of a few configurations before accepting it.
//
// Baseline history after the computation-pack refactor:
// - 2026-11-10 (math correctness review): two additions, identical in all 162 configurations and
//   nothing else. (1) `PHILOX_WGSL` gained `philoxOpenUnitFloat`, the open-interval uniform
//   (core-numerics-05); the permute kernel includes the module and does not call the function.
//   (2) The shared `getTwoSidedPValue` of `SPATIAL_AUTOCORRELATION_FLOAT_WGSL` gained the
//   constant `ANALYTIC_P_VALUE_FLOOR` and clamps to it instead of `min(..., 1.0)`
//   (spatial-statistics-08); the permute kernel includes the module and does not call the
//   function, so no permutation result changes.
// - 2026-11-16 (F-STATS): exact uint32 count helpers, runtime row parameters and safe scaled
//   quotients replace the float32-count implementations. The complete fixture is refreshed;
//   independent permutation oracles retain the numerical-result contract.
// - 2026-11-20 (test round t1): retain the WGS-15 storage-binding address references and S3
//   resident significance validation. Positive float32 encodings strictly between 0x00000000
//   and 0x3f800000 represent exactly the accepted (0, 1) interval, including subnormals.
//   The fixture changes only by these references, the validity helper and its row guard;
//   permutation draws and statistic arithmetic retain their previous baseline text.
it('preserves the complete permutation WGSL baseline byte for byte across 162 configurations', () => {
  const kernels: Record<string, string> = {};
  for (const statistic of [
    'localMoran',
    'localGeary',
    'localG',
    'localGStar',
    'localBivariateMoran',
    'localLeeL',
    'localJoinCount',
    'localBivariateJoinCount'
  ] as const) {
    for (const joinCase of statistic === 'localBivariateJoinCount'
      ? (['BJC', 'CLC'] as const)
      : [undefined]) {
      for (const alternative of ['two-sided', 'greater', 'lesser', 'directed', 'folded'] as const) {
        for (const islands of ['exclude', 'keep'] as const) {
          for (const stopAfterExceedances of alternative === 'folded'
            ? [undefined]
            : [undefined, 5]) {
            const graph = new GPUCommandGraph(createNullWebGPUDevice());
            const test = createReplayTest(graph, {
              statistic,
              joinCase,
              alternative,
              islands,
              stopAfterExceedances
            });
            const nodes = test.getCommandNodes(graph).filter(node => node.id === 'golden-permute');
            const key = [
              statistic,
              joinCase ?? '-',
              alternative,
              islands,
              stopAfterExceedances ?? '-'
            ].join('/');
            kernels[key] = getGPUCommandNodeKernelSources(nodes)[0].source;
            expect(findReservedIdentifiers(stripWGSLComments(kernels[key])), key).toEqual([]);
            expect(findStatisticsNonFiniteConstants(kernels[key]), key).toEqual([]);
            graph.device.destroy();
          }
        }
      }
    }
  }
  const fixture = new URL('./fixtures/local-permutation-kernels.json', import.meta.url);
  expect(Object.keys(kernels)).toHaveLength(162);
  if (process.env['UPDATE_LOCAL_PERMUTATION_KERNELS']) {
    writeFileSync(fixture, `${JSON.stringify(kernels, null, 2)}\n`);
  }
  expect(kernels).toEqual(JSON.parse(readFileSync(fixture, 'utf8')));
});
