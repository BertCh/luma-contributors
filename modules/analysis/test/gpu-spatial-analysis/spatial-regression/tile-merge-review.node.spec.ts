// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  useRegressionReductionTestHooks,
  getRegressionReductionNativeNodeDelta
} from './regression-reduction-test-hooks';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {getRegressionTraceCommandNodeCount} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-command-counts';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getRegressionTraceLayout} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-trace-layout';
import {getRowTileSumNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import * as allocationPlans from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-likelihood-allocations';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterLength} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {createGWRBatchMergeFixture} from './tile-merge-review-fixtures';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

function createFixture(rowCount: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', rowCount + 1),
    neighbors: view('uint32', rowCount),
    weights: view('float32', rowCount)
  };
  return {device, graph, view, weights};
}

useRegressionReductionTestHooks();

it('R2 GM bounds nested OLS/BP and optional transpose/trace commands', () => {
  for (const estimator of ['kelejian-prucha', 'homoskedastic'] as const) {
    const {device, graph, view, weights} = createFixture(65);
    try {
      const contributor = new GPUSpatialErrorGM({
        weights,
        estimator,
        predictors: view('float32', 65),
        response: view('float32', 65),
        predictorCount: 1,
        tileRowCount: 1,
        traceMode: 'stochastic',
        traceProbeCount: 3,
        output: {
          table: view('float32', 12),
          summary: view('float32', 13),
          status: {code: view('uint32', 1)}
        }
      });
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
    } finally {
      device.destroy();
    }
  }
});

it('R8 executes the real last-chunk norm producer and observes slot 4096', () => {
  const {device, graph, view} = createFixture(4097);
  try {
    const shared = {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 1),
      weights: view('float32', 1)
    };
    const tailWeights = {...shared, weights: view('float32', 1)};
    const contributor = new GPUSpatialWeightsTraces({
      id: 'real-tail',
      maximumPower: 1,
      mode: 'stochastic',
      probeCount: 1,
      tileRowCount: 32,
      weights: {
        rowCount: 4097,
        chunks: Array.from({length: 4097}, (_, rowStart) => ({
          rowStart,
          rowCount: 1,
          weights: rowStart === 4096 ? tailWeights : shared
        }))
      },
      output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    const tail = nodes.find(node => node.id === 'real-tail-norm-tiles-chunk4096')!;
    const group = nodes.find(node => node.id === 'real-tail-norm-merge-normPartial-groups')!;
    const merge = nodes.find(node => node.id === 'real-tail-norm-merge')!;
    const partial = group.resources!.filter(resource => 'buffer' in resource)[0]
      .buffer as GraphDataView;
    const total = merge.resources!.filter(resource => 'buffer' in resource)[1]
      .buffer as GraphDataView;
    const memory = createCPUGraphMemory();
    memory.write(shared.offsets, Uint32Array.of(0, 1));
    memory.write(shared.neighbors, Uint32Array.of(0));
    memory.write(shared.weights, Float32Array.of(1));
    memory.write(tailWeights.weights, Float32Array.of(7));
    executeCommandNodesOnCPU([tail, group, merge], memory);
    expect(memory.readFloat32(partial)[4096]).toBe(7);
    expect(memory.readFloat32(total)[0]).toBe(7);
  } finally {
    device.destroy();
  }
});

it('R8 advances the real GWR cursor into distinct spreads for runs 32 and 33', () => {
  const {device, graph, view} = createFixture(4097);
  try {
    const parameters = view('uint32', 4),
      spreads = view('float32', 68);
    const fixture = createGWRBatchMergeFixture(graph, parameters, spreads);
    const memory = createCPUGraphMemory();
    memory.write(parameters, getGPUPermutationParameterValues({seed: 42, permutationCount: 33}));
    executeCommandNodesOnCPU(fixture.nodes, memory);
    // 4096 balanced +/- a deviations: means 0, population spreads a and 2a, exactly.
    expect(Array.from(memory.readFloat32(spreads).slice(62, 68))).toEqual([1, 2, 3, 6, 5, 10]);
    expect(Array.from(memory.readFloat32(fixture.counts).slice(31, 34))).toEqual([
      4096, 4096, 4096
    ]);
    expect(memory.readFloat32(fixture.partial)[fixture.cursorSlot]).toBe(64);
  } finally {
    device.destroy();
  }
}, 20000);

it('R7 distinguishes the final cascade from a serial group merge at 12289 tiles', () => {
  const {device, graph, view} = createFixture(12289);
  try {
    const source = view('float32', 12289),
      total = view('float32', 1);
    const values = new Float32Array(12289);
    values[0] = 2 ** 25;
    values[8192] = -(2 ** 25);
    values[12288] = 1;
    const nodes = getRowTileSumNodes(graph, {
      id: 'four-groups',
      operation: 'FourGroupTest',
      variant: 'sum',
      rowCount: 12289,
      layout: {tileCount: 12289, tileRowCount: 1},
      sumCount: 1,
      bindings: [{name: 'source', view: source, type: 'f32', access: 'read'}],
      declarations:
        'fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) { (*terms)[0] += source[sourceOffset + row]; }',
      partial: view('float32', 12289),
      output: total
    });
    const memory = createCPUGraphMemory();
    memory.write(source, values);
    executeCommandNodesOnCPU(nodes, memory);
    // Float32 group totals [2^25, 0, -2^25, 1]: cascade=0, serial=1, float64=1.
    expect(memory.readFloat32(total)[0]).toBe(0);
    let serial = Math.fround(0);
    for (const value of [2 ** 25, 0, -(2 ** 25), 1]) serial = Math.fround(serial + value);
    expect(serial).toBe(1);
  } finally {
    device.destroy();
  }
});

it('R4 retains the initial trace width when the chunk count alone needs a hierarchy', () => {
  const {device, view} = createFixture(1052672);
  try {
    const single = {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 1),
      weights: view('float32', 1)
    };
    const chunks = Array.from({length: 4096}, (_, rowStart) => ({
      rowStart,
      rowCount: 1,
      weights: single
    }));
    chunks.push({
      rowStart: 4096,
      rowCount: 1048576,
      weights: {
        offsets: view('uint32', 1048577),
        neighbors: view('uint32', 1048576),
        weights: view('float32', 1048576)
      }
    });
    expect(getRegressionTraceLayout('uneven', {rowCount: 1052672, chunks}, 1052672)).toEqual({
      tileRowCount: 1028,
      tileCount: 5117
    });
  } finally {
    device.destroy();
  }
});

it('R5 preflights information-probe partial and group bindings using chunk-local tiles', () => {
  const {device, graph, view} = createFixture(8192);
  const planner = vi.spyOn(allocationPlans, 'getSpatialProbeAllocationLengths');
  try {
    const contributor = new GPUSpatialLagMaximumLikelihood({
      id: 'information',
      weights: {
        rowCount: 8192,
        chunks: [1, 8191].map((rowCount, index) => ({
          rowStart: index,
          rowCount,
          weights: {
            offsets: view('uint32', rowCount + 1),
            neighbors: view('uint32', rowCount),
            weights: view('float32', rowCount)
          }
        }))
      },
      predictors: view('float32', 8192),
      response: view('float32', 8192),
      predictorCount: 1,
      tileRowCount: 2,
      maximumPower: 2,
      traceMode: 'stochastic',
      traceProbeCount: 3,
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const call = planner.mock.calls.findIndex(
      arguments_ => arguments_[0] === 8192 && arguments_[3] === 1
    );
    const plan = planner.mock.results[call].value as Record<string, number>;
    expect(plan['probe partials']).toBe(12291);
    expect(plan['probe merge groups']).toBe(6);
    const group = nodes.find(node => node.id === 'information-tr3-probe-merge-0-partial-groups')!;
    const resources = group.resources!.filter(resource => 'buffer' in resource);
    expect((resources[0].buffer as GraphDataView).length).toBe(plan['probe partials']);
    expect((resources[1].buffer as GraphDataView).length).toBe(plan['probe merge groups']);
  } finally {
    planner.mockRestore();
    device.destroy();
  }
});

it('R6 sizes hierarchical GWR batches for four commands and preserves one-level batches', () => {
  for (const rows of [4096, 4097]) {
    const {device, view} = createFixture(rows);
    try {
      const contributor = new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions: view('float32x2', rows),
        predictors: view('float32', rows),
        response: view('float32', rows),
        predictorCount: 1,
        coefficients: view('float32', rows * 2),
        bandwidthParameters: view(
          'float32',
          getGPUGeographicallyWeightedRegressionParameterLength()
        ),
        selectedBandwidth: view('float32', 2),
        parameters: view('uint32', 4),
        maximumPermutations: 262143,
        tileRowCount: 1,
        output: {table: view('float32', 12), summary: view('float32', 4)}
      });
      expect(contributor.batchRunCount).toBe(rows === 4096 ? 32 : 35);
      expect(contributor.estimatedCommandNodeCount).toBe(rows === 4096 ? 26656 : 32040);
    } finally {
      device.destroy();
    }
  }
});

it('R3 preserves the first usable origin, integer count and centered moments across empty groups', () => {
  const {device, graph, view} = createFixture(8193);
  try {
    const predictors = view('float32', 8193),
      response = view('float32', 8193),
      mask = view('uint32', 8193);
    const contributor = new GPUOrdinaryLeastSquares({
      id: 'origin',
      predictors,
      response,
      mask,
      predictorCount: 1,
      tileRowCount: 1,
      output: {
        coefficients: view('float32', 2),
        standardErrors: view('float32', 2),
        tStatistics: view('float32', 2),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const last = nodes.findIndex(node => node.id === 'origin-moments-merge');
    // The trusted max primitive is serialized for the interpreter; compare its real
    // declaration count, not the smaller stand-in count, with the production estimate.
    expect(nodes.length + getRegressionReductionNativeNodeDelta()).toBe(
      contributor.estimatedCommandNodeCount
    );
    const memory = createCPUGraphMemory();
    memory.write(
      predictors,
      Float32Array.from({length: 8193}, (_, row) => 1000 + [-3, -1, 1, 3][row % 4])
    );
    memory.write(
      response,
      Float32Array.from(
        {length: 8193},
        (_, row) => 2005 + 2 * [-3, -1, 1, 3][row % 4] + [1, -1, -1, 1][row % 4]
      )
    );
    memory.write(
      mask,
      Uint32Array.from({length: 8193}, (_, row) => Number(row >= 4096 && row < 8192))
    );
    executeCommandNodesOnCPU(nodes.slice(0, last + 1), memory);
    const workspace = nodes[last].resources!.filter(resource => 'buffer' in resource)[1]
      .buffer as GraphDataView;
    const values = memory.readFloat32(workspace);
    expect(Array.from(values.slice(0, 2))).toEqual([1000, 2005]);
    expect(memory.readUint32(workspace)[2]).toBe(4096);
    const source = getGPUCommandNodeKernelSources([nodes[last]])[0].source;
    const momentSlot = Number(source.match(/const W_MOMENTS: u32 = (\d+)u/)![1]);
    // Four balanced rows: sum(x^2)=20, sum(x*y)=40, sum(y^2)=84; repeated 1024 times.
    expect(Array.from(values.slice(momentSlot, momentSlot + 3))).toEqual([20480, 40960, 86016]);
  } finally {
    device.destroy();
  }
}, 20000);

it('R2 counts group passes and refuses 100 single-probe batches before allocation', () => {
  for (const rows of [4096, 4097]) {
    const {device, graph, view, weights} = createFixture(rows);
    try {
      const props = {
        id: 'review-budget',
        weights,
        tileRowCount: 1,
        maximumPower: 128,
        mode: 'stochastic' as const,
        probeCount: 100,
        probeBatchSize: 1,
        output: {powerTraces: view('float32', 128), powerSums: view('float32', 128)}
      };
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(
        getRegressionTraceCommandNodeCount({
          weights,
          tileCount: rows,
          maximumPower: 128,
          probeTotal: 100,
          batchProbeCount: 1,
          stochastic: true
        })
      ).toBe(rows === 4096 ? 25734 : 38536);
      if (rows === 4097) {
        expect(() => new GPUSpatialWeightsTraces(props)).toThrow(
          /estimated command node count.*32768/
        );
        expect(allocate).not.toHaveBeenCalled();
      } else {
        const contributor = new GPUSpatialWeightsTraces(props);
        // Counted independently: 130 chunk passes, two merges, batch table, 100*(128+128), final.
        expect(
          (contributor as unknown as {estimatedCommandNodeCount: number}).estimatedCommandNodeCount
        ).toBe(25734);
        expect(allocate).not.toHaveBeenCalled();
        expect(contributor.getCommandNodes(graph)).toHaveLength(25734);
      }
    } finally {
      device.destroy();
    }
  }
});

for (const Contributor of [GPUSpatialLagMaximumLikelihood, GPUSpatialErrorMaximumLikelihood]) {
  it(`R2 ${Contributor.name} bounds its emitted graph including nested contributors`, () => {
    for (const chunks of [1, 3]) {
      const {device, graph, view, weights} = createFixture(65);
      try {
        const inputWeights =
          chunks === 1
            ? weights
            : {
                rowCount: 65,
                chunks: [1, 32, 32].map((rowCount, index) => ({
                  rowStart: [0, 1, 33][index],
                  rowCount,
                  weights: {
                    offsets: view('uint32', rowCount + 1),
                    neighbors: view('uint32', rowCount),
                    weights: view('float32', rowCount)
                  }
                }))
              };
        const contributor = new Contributor({
          weights: inputWeights,
          predictorCount: 1,
          predictors: view('float32', 65),
          response: view('float32', 65),
          tileRowCount: 1,
          maximumPower: 3,
          traceMode: 'stochastic',
          traceProbeCount: 3,
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          }
        });
        expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
          contributor.estimatedCommandNodeCount
        );
      } finally {
        device.destroy();
      }
    }
  });
  it(`R2 ${Contributor.name} refuses over-budget chunk dispatches at construction`, () => {
    const {device, graph, view} = createFixture(4097);
    try {
      const chunkWeights = {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      };
      const props = {
        id: 'review-chunk-budget',
        weights: {
          rowCount: 4097,
          chunks: Array.from({length: 4097}, (_, rowStart) => ({
            rowStart,
            rowCount: 1,
            weights: chunkWeights
          }))
        },
        predictorCount: 1,
        predictors: view('float32', 4097),
        response: view('float32', 4097),
        maximumPower: 2,
        traceMode: 'stochastic' as const,
        traceProbeCount: 1,
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)}
        }
      };
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => new Contributor(props)).toThrow(/estimated command node count.*32768/);
      expect(allocate).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
}
