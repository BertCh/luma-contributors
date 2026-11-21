// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {GPURipleyEnvelope} from '../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope';
import {GPUPeriodIndex} from '../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUCellPeriodKeys} from '../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {GPUValueCube} from '../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {GPUPointCloudNeighborhoods} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods';
import {GPUVariogram} from '../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {GPUSpaceTimeVariogram} from '../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {GPUSpatialCorrelogram} from '../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {GPURipleyEnvelopeOperation} from '../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation';
import {CPURipleyEnvelopeOperation} from '../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import type {GPUContributorOperationViewGetter} from '../../src/analysis-core/operations/gpu-contributor-operation';

for (const subject of ['envelope', 'periods', 'keys', 'cube'] as const) {
  it(`R1 ${subject} clears only its one scalar result despite trailing capacity`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let sequence = 0;
      const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x2'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `review-${sequence++}`, format, length);
      const statusStorage = view('uint32', 3);
      // Standard status ports require a one-row view, even when the backing
      // allocation has trailing capacity. The envelope uses its own count port.
      const status =
        subject === 'envelope'
          ? statusStorage
          : graph.createDataView(statusStorage.buffer, {format: 'uint32', length: 1});
      const contributor =
        subject === 'envelope'
          ? new GPURipleyEnvelope({
              positions: view('float32x2', 2),
              parameters: view('float32', 8),
              gridSize: [2, 2],
              radiusCount: 1,
              simulationCount: 2,
              output: {lower: view('float32', 1), unplacedCount: status}
            })
          : subject === 'periods'
            ? new GPUPeriodIndex({
                temporalContext: {
                  encoding: 'int64-words',
                  epoch: 'unix',
                  unit: 'millisecond',
                  axis: {kind: 'irregular'}
                },
                timestamps: view('uint32x2', 1),
                periods: {kind: 'fixed', origin: 0n, width: 1n, count: 1},
                output: {periods: view('uint32', 1), status: {invalidCount: status}}
              })
            : subject === 'keys'
              ? new GPUCellPeriodKeys({
                  cells: view('uint32x2', 1),
                  periods: view('uint32', 1),
                  periodCount: 2,
                  fill: 'zeros',
                  layout: 'legacy-one-bit',
                  output: {keys: view('uint32x2', 1), status: {invalidCount: status}}
                })
              : new GPUValueCube({
                  cells: view('uint32', 1),
                  periods: view('uint32', 1),
                  values: view('float32', 1),
                  cellCount: 1,
                  periodCount: 1,
                  output: {counts: view('uint32', 1), status: {invalidCount: status}}
                });
      const clear = contributor
        .getCommandNodes(graph)
        .find(node =>
          node.id.endsWith(subject === 'envelope' ? '-unplaced-clear' : '-clear-invalid-count')
        )!;
      // The contract has exactly one scalar output. Words 1 and 2 are
      // caller-owned trailing allocation, independently of the status view length.
      expect(clear.workload?.maximumInvocationCount).toBe(1);
    } finally {
      device.destroy();
    }
  });
}

it('R5 both envelope operation forms reject three resolved rows when rowCount is two', () => {
  const props = {
    positions: 'positions',
    parameters: 'parameters',
    rowCount: 2,
    gridSize: [2, 2] as const,
    radiusCount: 1,
    simulationCount: 2,
    output: {lower: 'lower', unplacedCount: 'unplaced'}
  };
  const positions = new Float32Array([0, 0, 1, 1, 0.5, 0.5]);
  const parameters = getGPURipleyParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 2});
  const cpu = new CPURipleyEnvelopeOperation(props);
  expect(() =>
    cpu.execute({
      inputs: {
        positions: {
          format: 'float32x2',
          length: 3,
          chunks: [{values: positions, length: 3}],
          pack: () => positions
        },
        parameters: {
          format: 'float32',
          length: parameters.length,
          chunks: [{values: parameters, length: parameters.length}],
          pack: () => parameters
        }
      }
    })
  ).toThrow(/positions.*rowCount/);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = {
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      lower: createTransientView(graph, 'lower', 'float32', 1),
      unplacedCount: createTransientView(graph, 'unplaced', 'uint32', 1)
    };
    const getView: GPUContributorOperationViewGetter = port =>
      views[port as keyof typeof views] as never;
    const gpu = new GPURipleyEnvelopeOperation(props);
    expect(() => gpu.createContributor(getView)).toThrow(/positions.*rowCount/);
  } finally {
    device.destroy();
  }
});

for (const Operation of [CPURipleyEnvelopeOperation, GPURipleyEnvelopeOperation]) {
  for (const getter of [false, true]) {
    it(`review 2 R2 ${Operation.name} snapshots caller rowCount once, getter=${getter}`, () => {
      let reads = 0;
      const props = {
        positions: 'positions',
        parameters: 'parameters',
        rowCount: 2,
        gridSize: [2, 2] as const,
        radiusCount: 1,
        simulationCount: 2,
        output: {lower: 'lower'}
      };
      if (getter)
        Object.defineProperty(props, 'rowCount', {
          get() {
            if (++reads > 1) throw new Error('rowCount getter reread');
            return 2;
          }
        });
      const operation = new Operation(props);
      if (!getter) props.rowCount = 3;
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const parameters = getGPURipleyParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 2});
        const resolve = (length: number) => {
          if (operation instanceof GPURipleyEnvelopeOperation) {
            const views = {
              positions: createTransientView(graph, `positions-${length}`, 'float32x2', length),
              parameters: createTransientView(graph, `parameters-${length}`, 'float32', 8),
              lower: createTransientView(graph, `lower-${length}`, 'float32', 1)
            };
            return operation.createContributor(port => views[port as keyof typeof views] as never);
          }
          const positions = new Float32Array(length * 2);
          return operation.execute({
            inputs: {
              positions: {
                format: 'float32x2',
                length,
                chunks: [{values: positions, length}],
                pack: () => positions
              },
              parameters: {
                format: 'float32',
                length: 8,
                chunks: [{values: parameters, length: 8}],
                pack: () => parameters
              }
            }
          });
        };
        expect(() => resolve(2)).not.toThrow();
        expect(() => resolve(3)).toThrow(/positions.*rowCount/);
        if (getter) expect(reads).toBe(1);
      } finally {
        device.destroy();
      }
    });
  }
}

for (const subject of ['variogram', 'correlogram', 'space-time'] as const) {
  it(`R4 ${subject} generates reciprocal-free encoding and decoding for tiny units`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let sequence = 0;
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `pair-${sequence++}`, format, length);
      const props = {
        positions: view('float32x2', 5),
        values: view('float32', 5),
        parameters: view('float32', 8),
        gridSize: [2, 2] as const
      };
      const contributor =
        subject === 'variogram'
          ? new GPUVariogram({...props, lagCount: 1, output: {semivariances: view('float32', 1)}})
          : subject === 'space-time'
            ? new GPUSpaceTimeVariogram({
                ...props,
                times: view('float32', 5),
                lagCount: 1,
                temporalLagCount: 1,
                output: {
                  semivariances: view('float32', 4),
                  pairCounts: view('uint32', 4),
                  status: view('uint32', 4)
                }
              })
            : new GPUSpatialCorrelogram({
                ...props,
                bandCount: 1,
                output: {moransI: view('float32', 1)}
              });
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      const generated = [...sources.values()].join('\n');
      // A variance 4/25 * 2^-94 with zero IQR gives u=variance/4096.
      // 2^24/u overflows, although every normalized pair term fits.
      expect(generated.includes('fn scalePairStatisticsAmount(')).toBe(true);
      expect(generated.includes('fn decodePairStatisticsAmount(')).toBe(true);
      // 2^-66 * [0,1,2,3,524288] has normal variance but subnormal
      // adaptive u. A separate exponent must survive storage on FTZ devices.
      expect(generated.includes('unitVarianceOffset + 1u] = unitPower')).toBe(true);
      if (subject === 'correlogram') {
        expect(generated.includes('fn getCorrelogramMomentPower(')).toBe(true);
        expect(generated.includes('normalizedSumOfSquares * normalizedSumOfSquares')).toBe(true);
      }
    } finally {
      device.destroy();
    }
  });
}

for (const runtimeBounds of [false, true]) {
  it(`review 2 R1 missing rows bypass support but real outside-grid rows retain it: runtime=${runtimeBounds}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let sequence = 0;
      const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `neighborhood-${sequence++}`, format, length);
      const nodes = new GPUPointCloudNeighborhoods({
        positions: view('float32x3', 4),
        queryRows: view('uint32', 1),
        radius: 1,
        capacity: 1,
        supportBounds: {minimum: [-1, -2, -2], maximum: [2, 2, 2]},
        parameters: runtimeBounds ? view('float32', 9) : undefined,
        output: {
          indices: view('uint32', 1),
          squaredDistances: view('float32', 1),
          counts: view('uint32', 1),
          status: view('uint32', 1)
        }
      }).getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      const status = sources.get(
        runtimeBounds ? 'gpu-pointcloud-neighborhoods-status' : 'gpu-pointcloud-neighborhoods'
      )!;
      // A missing row has no sphere: invalid bit 1 must be selected
      // before any support test can inherit the clamped source point.
      expect(status).toMatch(/if\s*\(queryRow >= 4u\)\s*\{\s*writeStatus\(index,\s*1u\)/);
      expect(status).toMatch(/select\(1u,\s*0u,\s*validQuery\)/);
    } finally {
      device.destroy();
    }
  });
}
