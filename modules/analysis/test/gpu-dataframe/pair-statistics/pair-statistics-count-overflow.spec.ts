// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {getPairStatisticsSegmentTotalNodes} from '../../../src/dataframe-analysis/pair-statistics/pair-statistics-cell-neighbors';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {getPairCountOverflowPartials} from './pair-count-overflow-injection';

for (const estimator of ['cross', 'marked'] as const) {
  it(`${estimator} pair-count overflow saturates counts and invalidates estimates`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    recordWGSLKernelNodes(rig.graph);
    const rows = estimator === 'cross' ? 92683 : 65537;
    const bins = 3;
    const positions = createTransientView(rig.graph, 'positions', 'float32x2', rows);
    const estimates = rig.output('float32', bins * (estimator === 'cross' ? 1 : 4));
    const pairCounts = rig.output('uint32', bins);
    const status = rig.output('uint32', bins);
    try {
      const contributor =
        estimator === 'cross'
          ? new GPUCrossVariogram({
              id: 'cross',
              positions,
              values: createTransientView(rig.graph, 'values', 'float32', rows),
              parameters: rig.input(Float32Array.from([1]), 'float32'),
              lagCount: bins,
              output: {semivariances: estimates, pairCounts, status}
            })
          : new GPUMarkedPointStatistics({
              id: 'marked',
              positions,
              parameters: rig.input(Float32Array.from([0, 0, 1, 1, 1, 0]), 'float32'),
              radiusCount: bins,
              edgeCorrection: 'none',
              output: {statistics: estimates, pairCounts, status}
            });
      const nodes = contributor.getCommandNodes(rig.graph);
      const countNodes = nodes.filter(node => node.workload?.variant === 'segment-total-u32');
      const finish = nodes.find(
        node => node.workload?.variant === (estimator === 'cross' ? 'legacy-finish' : 'grid-finish')
      )!;
      const finishRecord = getWGSLKernelNodeRecord(rig.graph, finish.id)!;
      // Runtime-parameter kernels declare resources directly rather than recording WGSL props.
      const partials = getPairCountOverflowPartials(countNodes[0]);
      const sums = finishRecord.bindings.find(binding => binding.name === 'totalSums')!.view;
      const moments = finishRecord.bindings.find(binding => binding.name === 'moments')?.view;
      // Inject the exact per-focus counts for coincident points, then execute the contributor's
      // real count-reduction tree and finish kernel. This replaces billions of candidate visits.
      const inject = createWGSLKernelNode(rig.graph, {
        id: 'synthetic-partials',
        operation: 'PairCountOverflowTest',
        invocationCount: rows * bins,
        bindings: [
          {name: 'partials', view: partials, type: 'u32', access: 'read_write'},
          {name: 'sums', view: sums, type: 'f32', access: 'read_write'},
          ...(moments
            ? [
                {
                  name: 'moments',
                  view: moments,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `let bin = index / ${rows}u; let row = index % ${rows}u;
var count = 0u;
if (bin == 0u) { count = ${estimator === 'cross' ? `${rows - 1}u - row` : `${rows - 1}u`}; }
if (bin == 1u && row == 0u) { count = 2u; }
partials[partialsOffset + index] = count;
if (index < ${sums.length}u) { sums[sumsOffset + index] = 1.0; }
${moments ? `if (index < 4u) { moments[momentsOffset + index] = select(1.0, ${rows}.0, index < 2u); }` : ''}`
      });
      rig.run({getCommandNodes: () => [inject, ...countNodes, finish]});
      expect(await rig.readUint(pairCounts)).toEqual([0xffffffff, 2, 0]);
      expect(await rig.readUint(status)).toEqual(estimator === 'cross' ? [3, 0, 1] : [3, 0, 0]);
      const actual = await rig.readFloat(estimates);
      expect(actual.slice(0, estimator === 'cross' ? 1 : 4).every(Number.isNaN)).toBe(true);
      if (estimator === 'cross') {
        expect(actual[1]).toBe(0.5);
        expect(actual[2]).toBeNaN();
      } else {
        expect(Number.isFinite(actual[4])).toBe(true);
      }
    } finally {
      rig.destroy();
    }
  });
}

it('count totals distinguish exact uint32 maximum from overflow and propagate overflow across levels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  // 257 rows force a second level. Each first-level overflow must survive subsequent additions.
  const rows = 257;
  const partials = new Uint32Array(rows * 5);
  partials.set([0xfffffffe, 1], 0);
  partials.set([0xffffffff, 1], rows);
  partials.set([0xffffffff, 2], rows * 2);
  partials[rows * 3] = 0xffffffff;
  partials[rows * 4 - 1] = 1;
  const totals = rig.output('uint32', 10);
  try {
    rig.run({
      getCommandNodes: graph =>
        getPairStatisticsSegmentTotalNodes(graph, {
          id: 'counts',
          operation: 'PairCountOverflowTest',
          segmentCount: 5,
          rowCount: rows,
          input: rig.input(partials, 'uint32'),
          output: totals,
          detectOverflow: true
        })
    });
    expect(await rig.readUint(totals)).toEqual([
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 1, 1, 1, 0
    ]);
  } finally {
    rig.destroy();
  }
});
