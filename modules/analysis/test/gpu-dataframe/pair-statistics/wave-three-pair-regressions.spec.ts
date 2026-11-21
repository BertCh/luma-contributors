// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createPairStatisticsHarness} from './pair-statistics-harness';
import {GPUVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {GPUSpatialCorrelogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {GPURipleyEnvelope} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope';
import {getGPUVariogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/variogram-parameters';
import {getGPUSpatialCorrelogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/spatial-correlogram-parameters';
import {getGPUPointPatternIndicesParameterValues} from '../../../src/dataframe-analysis/pair-statistics/point-pattern-indices-parameters';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';

it('W2-11 excluded high rows preserve semivariance one half and centered product two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const correlogram of [false, true]) {
    for (const excluded of [0, 10]) {
      // BC1 included values [1,2,-1,-2,3,-3,4,-4,2e6,-2e6] have mean 0. Only the first
      // two points are close. Variogram=(2-1)^2/2=1/2; centered product=1*2=2.
      const values = Float32Array.from([
        1,
        2,
        -1,
        -2,
        3,
        -3,
        4,
        -4,
        2e6,
        -2e6,
        ...Array<number>(excluded).fill(1e9)
      ]);
      const positions = Float32Array.from([
        0,
        0,
        1,
        0,
        ...Array.from({length: 8}, (_, index) => [10 * (index + 1), 10 * (index + 1)]).flat(),
        ...Array.from({length: excluded}, () => [200, 200]).flat()
      ]);
      const harness = createPairStatisticsHarness(device, {
        positions,
        values,
        parameterLength: 8,
        outputs: {
          result: {format: 'float32', length: 1},
          pairs: {format: 'uint32', length: 1},
          overflow: {format: 'uint32', length: 1}
        },
        createContributor: views => {
          const props = {
            positions: views.positions,
            values: views.values!,
            parameters: views.parameters,
            gridSize: [8, 8] as const
          };
          return correlogram
            ? new GPUSpatialCorrelogram({
                ...props,
                bandCount: 1,
                output: {
                  moransI: views.outputs.result as never,
                  pairCounts: views.outputs.pairs as never,
                  status: {overflow: views.outputs.overflow as never}
                }
              })
            : new GPUVariogram({
                ...props,
                lagCount: 1,
                output: {
                  semivariances: views.outputs.result as never,
                  pairCounts: views.outputs.pairs as never,
                  status: {overflow: views.outputs.overflow as never}
                }
              });
        }
      });
      try {
        const settings = {bounds: [0, 0, 100, 100] as const, maximumDistance: 1.1};
        const result = await harness.run(
          correlogram
            ? getGPUSpatialCorrelogramParameterValues(settings)
            : getGPUVariogramParameterValues(settings)
        );
        expect(result['pairs']).toEqual([1]);
        expect(result['overflow']).toEqual([0]);
        expect(Number.isFinite(result['result'][0])).toBe(true);
        // Moran I = n * product / M2, with n=10 and M2=8e12+60.
        const product = correlogram
          ? (result['result'][0] * (8e12 + 60)) / 10
          : result['result'][0];
        expect(product).toBeCloseTo(correlogram ? 2 : 0.5, 4);
      } finally {
        harness.destroy();
      }
    }
  }
});

it('WGS-07 assigns the extreme point to column six', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.of(3 * 2 ** 125, 0.5),
    parameterLength: 8,
    outputs: {counts: {format: 'uint32', length: 8}},
    createContributor: views =>
      new GPUPointPatternIndices({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [8, 1],
        quadratGrid: [8, 1],
        output: {quadratCounts: views.outputs.counts as never}
      })
  });
  try {
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 2 ** 127, 1], indexCellSizeHint: 1})
    );
    expect(result['counts']).toEqual([0, 0, 0, 0, 0, 0, 1, 0]);
  } finally {
    harness.destroy();
  }
});

it('NAM-11 index cell hints preserve nearest distances beyond the hint', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.of(0, 0, 3, 0),
    parameterLength: 8,
    outputs: {distances: {format: 'float32', length: 2}},
    createContributor: views =>
      new GPUPointPatternIndices({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [4, 1],
        output: {nearestNeighborDistances: views.outputs.distances as never}
      })
  });
  try {
    for (const indexCellSizeHint of [1, 4]) {
      const result = await harness.run(
        getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 4, 1], indexCellSizeHint})
      );
      expect(result['distances']).toEqual([3, 3]);
    }
  } finally {
    harness.destroy();
  }
});

it('NUM-04 publishes no probability for zero usable border simulations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const observed = createInputBuffer(device, Float32Array.of(0.5));
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.of(0.5, 0.5, 0.51, 0.5),
    parameterLength: 8,
    outputs: {
      statistics: {format: 'float32', length: 6},
      lower: {format: 'float32', length: 1},
      upper: {format: 'float32', length: 1}
    },
    createContributor: views => ({
      getCommandNodes: graph =>
        new GPURipleyEnvelope({
          positions: views.positions,
          parameters: views.parameters,
          observedK: importGraphBuffer(graph, 'observed', observed, 'float32', 1),
          gridSize: [2, 2],
          radiusCount: 1,
          simulationCount: 2,
          output: {
            statistics: views.outputs.statistics as never,
            lower: views.outputs.lower as never,
            upper: views.outputs.upper as never
          }
        }).getCommandNodes(graph)
    })
  });
  try {
    const result = await harness.run(
      getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 0.45,
        edgeCorrection: 'border',
        seed: 0
      })
    );
    expect(result['statistics'][1]).toBe(0);
    expect(Number.isFinite(result['statistics'][2])).toBe(true);
    expect(result['statistics'][2]).toBeCloseTo(0.0510577075, 6);
    expect(Number.isNaN(result['statistics'][4])).toBe(true);
    expect(result['lower']).toHaveLength(1);
    expect(result['lower'].every(Number.isNaN)).toBe(true);
    expect(result['upper']).toHaveLength(1);
    expect(result['upper'].every(Number.isNaN)).toBe(true);
  } finally {
    harness.destroy();
    observed.destroy();
  }
});

it('W2-11 overflow reduction forwards a distant saturated bin and histogram loss and clears', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Maximum public bin count, spanning two 256-lane kernel workgroups.
  const binCount = 512;
  for (const loss of ['count', 'histogram', 'none'] as const) {
    const rig = new AnalysisRig(device);
    recordWGSLKernelNodes(rig.graph);
    try {
      const overflow = rig.output('uint32', 1);
      const nodes = new GPUVariogram({
        id: 'overflow-probe',
        positions: createTransientView(rig.graph, 'positions', 'float32x2', 2),
        values: createTransientView(rig.graph, 'values', 'float32', 2),
        parameters: createTransientView(rig.graph, 'parameters', 'float32', 8),
        gridSize: [1, 1],
        lagCount: binCount,
        output: {
          semivariances: createTransientView(rig.graph, 'result', 'float32', binCount),
          status: {overflow}
        }
      }).getCommandNodes(rig.graph);
      const reduction = nodes.filter(node => node.id.startsWith('overflow-probe-overflow'));
      const record = getWGSLKernelNodeRecord(rig.graph, reduction[0].id)!;
      const accumulators = record.bindings.find(binding => binding.name === 'accumulators')!.view;
      const inject = createWGSLKernelNode(rig.graph, {
        id: 'inject-loss',
        operation: 'PairOverflowProbe',
        invocationCount: accumulators.length,
        bindings: [
          {name: 'words', view: accumulators, type: 'u32', access: 'read_write'},
          {name: 'previousOverflow', view: overflow, type: 'u32', access: 'read_write'}
        ],
        body: `words[wordsOffset + index] = select(0u, ${loss === 'count' ? '0xffffffffu' : '1u'}, index == ${loss === 'count' ? (binCount - 1) * 8 : accumulators.length - 1}u && ${loss === 'none' ? 'false' : 'true'});
if (index == 0u) { previousOverflow[previousOverflowOffset] = 1u; }`
      });
      rig.run({getCommandNodes: () => [inject, ...reduction]});
      expect(await rig.readUint(overflow)).toEqual([loss === 'none' ? 0 : 1]);
    } finally {
      rig.destroy();
    }
  }
});

it('S2 huge cell-size hints preserve unit-frame nearest-neighbor distances', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsHarness(device, {
    positions: Float32Array.of(0, 0, 1, 0),
    parameterLength: 8,
    outputs: {distances: {format: 'float32', length: 2}},
    createContributor: views =>
      new GPUPointPatternIndices({
        positions: views.positions,
        parameters: views.parameters,
        gridSize: [2, 2],
        output: {nearestNeighborDistances: views.outputs.distances as never}
      })
  });
  try {
    for (const indexCellSizeHint of [1, 2 ** 66, Math.fround(3.4e38)]) {
      const result = await harness.run(
        getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 1, 1], indexCellSizeHint})
      );
      expect(result['distances']).toEqual([1, 1]);
    }
  } finally {
    harness.destroy();
  }
});
