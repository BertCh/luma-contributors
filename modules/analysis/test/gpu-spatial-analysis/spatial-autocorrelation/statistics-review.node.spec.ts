// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalGeary,
  GPUSpatialPearson,
  GPUGammaStatistic
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPULocalPermutationTest,
  GPUGlobalPermutationTest
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  GPUMultipleTestingAdjustment,
  GPURasterFalseDiscoveryRateMask
} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {GPULocalPermutationReplay} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getNegativeValueCountNodes} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createScene(rows = 5, chunkSizes: readonly number[] = [rows]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  let rowStart = 0;
  const chunks = chunkSizes.map(rowCount => {
    const result = {
      rowStart,
      rowCount,
      weights: {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', rowCount),
        weights: view('float32', rowCount)
      }
    };
    rowStart += rowCount;
    return result;
  });
  return {device, graph, view, weights: {rowCount: rows, chunks}, values: view('float32', rows)};
}

it('Local Geary rejects the same view and aligned-prefix aliases across every output', () => {
  const {device, graph, view, weights, values} = createScene();
  try {
    const statistic = view('float32', 10);
    const result = graph.createDataView(statistic.buffer, {format: 'float32', length: 5});
    for (const globalStatistics of [
      result,
      graph.createDataView(statistic.buffer, {format: 'float32', length: 5, byteOffset: 20})
    ]) {
      expect(
        () =>
          new GPULocalGeary({
            values,
            weights,
            parameters: view('float32', 8),
            output: {statistic: result, globalStatistics}
          })
      ).toThrow(/overlap/);
    }
  } finally {
    device.destroy();
  }
});

for (const Contributor of [GPUHotSpotAnalysis, GPULocalMoran, GPULocalGeary]) {
  it(`${Contributor.name} rejects CSR buffers without STORAGE before any resource declaration`, () => {
    const {device, graph, view, values} = createScene();
    const neighborsBuffer = device.createBuffer({byteLength: 20, usage: Buffer.COPY_DST});
    try {
      const weights = {
        offsets: view('uint32', 6),
        weights: view('float32', 5),
        neighbors: importGraphBuffer(graph, 'invalid-usage', neighborsBuffer, 'uint32', 5)
      };
      const parameters = view('float32', 8);
      const output = {zScores: view('float32', 5), statistic: view('float32', 5)};
      const contributor = new Contributor({values, weights, parameters, output});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
    } finally {
      neighborsBuffer.destroy();
      device.destroy();
    }
  });
}

for (const local of [false, true]) {
  it(`${local ? 'local' : 'global'} permutation rejects missing STORAGE usage before declarations`, () => {
    const {device, graph, view, values} = createScene();
    const neighborsBuffer = device.createBuffer({byteLength: 20, usage: Buffer.COPY_DST});
    try {
      const weights = {
        offsets: view('uint32', 6),
        weights: view('float32', 5),
        neighbors: importGraphBuffer(graph, 'invalid-usage', neighborsBuffer, 'uint32', 5)
      };
      const common = {values, weights, parameters: view('uint32', 4), maximumPermutations: 3};
      const contributor = local
        ? new GPULocalPermutationTest({
            ...common,
            statistic: 'localMoran',
            output: {
              exceedances: view('uint32', 5),
              pseudoPValues: view('float32', 5),
              status: {overflow: view('uint32', 1)}
            }
          })
        : new GPUGlobalPermutationTest({
            ...common,
            statistic: 'moran',
            output: {summary: view('float32', 12)}
          });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
    } finally {
      neighborsBuffer.destroy();
      device.destroy();
    }
  });
}

it('permutation and multiple-testing contributors reject writable aliases at construction', () => {
  const {device, graph, view, weights, values} = createScene();
  try {
    const floats = view('float32', 16);
    const first = graph.createDataView(floats.buffer, {format: 'float32', length: 5});
    const next = graph.createDataView(floats.buffer, {
      format: 'float32',
      length: 5,
      byteOffset: 20
    });
    const integer = graph.createDataView(floats.buffer, {format: 'uint32', length: 5});
    expect(
      () =>
        new GPULocalPermutationTest({
          values,
          weights,
          parameters: view('uint32', 4),
          maximumPermutations: 3,
          statistic: 'localMoran',
          output: {exceedances: integer, pseudoPValues: next, status: {overflow: view('uint32', 1)}}
        })
    ).toThrow(/overlap/);
    const summary = view('float32', 12);
    expect(
      () =>
        new GPUGlobalPermutationTest({
          values,
          weights,
          parameters: view('uint32', 4),
          maximumPermutations: 3,
          statistic: 'moran',
          output: {
            summary,
            referenceDistribution: graph.createDataView(summary.buffer, {
              format: 'float32',
              length: 3
            })
          }
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUMultipleTestingAdjustment({
          pValues: values,
          method: 'holm',
          output: {
            adjustedPValues: first,
            effectiveFamilySize: graph.createDataView(first.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: 20
            })
          }
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPURasterFalseDiscoveryRateMask({
          pValues: values,
          cellCount: 5,
          parameters: view('float32', 1),
          output: {significant: integer, adjustedPValues: next}
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('negative diagnostics remain uint32 through their reduction and publication', () => {
  const {device, graph, view} = createScene(2 ** 24 + 1);
  try {
    const {negativeCount, nodes} = getNegativeValueCountNodes(graph, {
      id: 'negative',
      operation: 'review',
      rows: 2 ** 24 + 1,
      compactX: view('float32', 2 ** 24 + 1),
      rowPositions: view('uint32', 2 ** 24 + 2)
    });
    expect(negativeCount.format).toBe('uint32');
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.find(source => source.id === 'negative-negative-indicators')?.source).toContain(
      'indicators: array<u32>'
    );
  } finally {
    device.destroy();
  }
});

for (const route of ['inline', 'materialized', 'sequential', 'materialized-sequential'] as const) {
  it(`global ${route} keeps pipeline count constant across unequal chunks and dispatches each chunk at its size`, () => {
    const build = (sizes: number[]) => {
      const rows = sizes.reduce((total, length) => total + length, 0);
      const {device, graph, view, weights, values} = createScene(rows, sizes);
      try {
        const contributor = new GPUGlobalPermutationTest({
          values,
          weights,
          parameters: view('uint32', 4),
          statistic: 'moran',
          maximumPermutations: 15,
          permutationBatchSize: route.includes('materialized') ? 3 : undefined,
          stopAfterExceedances: route.includes('sequential') ? 3 : undefined,
          output: {summary: view('float32', 12)}
        });
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        const pairs = nodes.filter(
          node => /-pairs(?:-chunk|$)/.test(node.id) || /-pairs-observed-chunk/.test(node.id)
        );
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const source of sources)
          expect(findReservedIdentifiers(stripWGSLComments(source.source)), source.id).toEqual([]);
        for (const source of sources.filter(source =>
          /(?:-pairs(?:-|$)|-materialize(?:-observed)?$)/.test(source.id)
        )) {
          expect(source.constants).toEqual({});
          if (source.source.includes('swapOrNotKeys'))
            expect(source.source).toMatch(/swapOrNotKeys: array<u32, 198>/);
        }
        return {
          pipelines: getGPUCommandNodePipelineKeys(pairs).size,
          work: pairs.map(
            node =>
              (node as {workload: {maximumInvocationCount: number}}).workload.maximumInvocationCount
          )
        };
      } finally {
        device.destroy();
      }
    };
    const few = build([6145, 6145]);
    const many = build([1, 2048, 2049, 8192]);
    expect(many.pipelines).toBe(few.pipelines);
    expect(new Set(many.work).size).toBeGreaterThan(1);
  });
}

it('30M-row local Geary classes use separate row-sized values and uint32 finite-count scratch', () => {
  const {device, graph, view, weights, values} = createScene(30_000_000);
  try {
    const nodes = new GPULocalPermutationTest({
      values,
      weights,
      statistic: 'localGeary',
      parameters: view('uint32', 4),
      maximumPermutations: 1,
      maximumNeighbors: 1,
      output: {
        exceedances: view('uint32', values.length),
        pseudoPValues: view('float32', values.length),
        classes: view('uint32', values.length),
        status: {overflow: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.find(({id}) => id.endsWith('-geary-mean-inputs'))!.source).toContain(
      'finiteFlags[finiteFlagsOffset + index] = u32(finite)'
    );
    expect(sources.find(({id}) => id.endsWith('-geary-classes'))!.source).toContain(
      'divideBySpatialCount(bitcast<f32>(sums[sumsOffset]), sums[sumsOffset + 1u])'
    );
    for (const {id, source} of sources)
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
  } finally {
    device.destroy();
  }
});

for (const Contributor of [GPUSpatialPearson, GPUGammaStatistic]) {
  it(`${Contributor.name} uses fixed sources for unequal chunks and rejects usage before allocation`, () => {
    const build = (sizes: number[]) => {
      const scene = createScene(2048, sizes);
      const nodes = new Contributor({
        values: scene.values,
        secondValues: scene.view('float32', 2048),
        weights: scene.weights,
        output: {statistic: scene.view('float32', 1)}
      }).getCommandNodes(scene.graph);
      return {...scene, nodes};
    };
    const first = build([1000, 1048]);
    const second = build([1, 255, 256, 1536]);
    try {
      expect(getGPUCommandNodePipelineKeys(first.nodes).size).toBe(4);
      expect(getGPUCommandNodePipelineKeys(second.nodes).size).toBe(4);
      for (const stage of ['-anchors', '-moments', '-centered']) {
        const nodes = second.nodes.filter(node => node.id.includes(stage));
        expect(
          nodes.map(
            node =>
              (node as {workload: {maximumWorkgroupCount: number}}).workload.maximumWorkgroupCount
          )
        ).toEqual([1, 1, 1, 6]);
        expect(new Set(getGPUCommandNodeKernelSources(nodes).map(({source}) => source)).size).toBe(
          1
        );
      }
      for (const {id, source} of getGPUCommandNodeKernelSources(second.nodes))
        expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
    } finally {
      first.device.destroy();
      second.device.destroy();
    }
    const {device, graph, view, values, weights} = createScene();
    const buffer = device.createBuffer({byteLength: 20, usage: Buffer.COPY_DST});
    try {
      const contributor = new Contributor({
        values,
        secondValues: view('float32', 5),
        weights: {
          ...weights.chunks[0].weights,
          neighbors: importGraphBuffer(graph, 'bad-edge-neighbors', buffer, 'uint32', 5)
        },
        output: {statistic: view('float32', 1)}
      });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
    } finally {
      buffer.destroy();
      device.destroy();
    }
  });
}

it('local permutation replay rejects overlapping distribution and metadata outputs at construction', () => {
  const {device, view, weights, values} = createScene();
  try {
    const test = new GPULocalPermutationTest({
      values,
      weights,
      statistic: 'localMoran',
      maximumPermutations: 3,
      parameters: view('uint32', 4),
      output: {
        exceedances: view('uint32', 5),
        pseudoPValues: view('float32', 5),
        status: {overflow: view('uint32', 1)}
      }
    });
    const overlapping = view('float32', 3);
    expect(
      () =>
        new GPULocalPermutationReplay({
          test,
          row: view('uint32', 1),
          output: {nullDistribution: overlapping, replayInfo: overlapping}
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('fixed moments retain the legacy layout and explicitly select extended odd counts', () => {
  const legacy = getGPUSpatialAutocorrelationParameterValues({
    fixedMoments: {count: 100, mean: 2, variance: 4}
  });
  expect(Array.from(legacy).slice(1)).toEqual([1, 100, 2, 4, 0, 0, 0]);
  legacy[2] = 200;
  expect(legacy[7]).toBe(0);
  for (const count of [2 ** 24 + 1, 30_000_001, 0xffffffff]) {
    const extended = getGPUSpatialAutocorrelationParameterValues({
      fixedMoments: {count, mean: 2, variance: 4}
    });
    expect(extended[7]).toBe(1);
    expect(extended[5] * 65536 + extended[6]).toBe(count);
    getGPUSpatialAutocorrelationParameterValues(
      {fixedMoments: {count: 100, mean: 2, variance: 4}},
      extended
    );
    expect(Array.from(extended).slice(5)).toEqual([0, 0, 0]);
  }
});
