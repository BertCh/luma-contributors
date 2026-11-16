// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  findStatisticsNonFiniteConstants,
  findStatisticsBarrierRisks
} from '../spatial-weights/statistics-wgsl-audit';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect, WgslExec, WgslParser} from 'wgsl_reflect';
import {SCAN_DRAW_INTERVAL_WGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-draw-interval';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import type {GPUSpatialScanModel} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createScene(model: GPUSpatialScanModel = 'bernoulli', maximumPermutations = 3) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scan-review-${serial++}`, format, length);
  const props = {
    model,
    positions: view('float32x2', 3),
    cases: view('uint32', 3),
    ...(model === 'poisson'
      ? {baseline: view('float32', 3)}
      : model === 'bernoulli'
        ? {controls: view('uint32', 3)}
        : {}),
    parameters: view('uint32', 8),
    timeBuckets: 1,
    maximumPermutations,
    maximumWindowZones: 3,
    maximumClusters: 1,
    batchReplicates: 1,
    output: {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', maximumPermutations + 1),
      summary: view('uint32', 4),
      status: {invalidCount: view('uint32', 1)}
    }
  };
  return {device, graph, view, props};
}

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const)
  it(`${model} uses saturated counts and audits every generated identifier`, () => {
    const {device, graph, props} = createScene(model);
    try {
      const nodes = new GPUSpatialScanStatistic(props).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(nodes.some(node => node.id.endsWith('invalid-count-status'))).toBe(true);
      expect(sources.find(item => item.id.endsWith('-totals'))?.source).toContain(
        'getSaturatedScanSum'
      );
      if (model !== 'space-time-permutation')
        expect(nodes.some(node => node.id.endsWith('cumulative-saturate'))).toBe(true);
      for (const {source} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
        if (/workgroupBarrier|storageBarrier/.test(source))
          expect(findStatisticsBarrierRisks(source)).toEqual([]);
        expect(() => new WgslReflect(source)).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });

it('scan rejects invalid caller usage before declarations and rejects writable aliases', () => {
  const {device, graph, props} = createScene();
  const buffer = device.createBuffer({byteLength: 12, usage: Buffer.COPY_DST});
  try {
    props.cases = importGraphBuffer(graph, 'invalid-cases', buffer, 'uint32', 3);
    const contributor = new GPUSpatialScanStatistic(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declare = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declare).not.toHaveBeenCalled();
    props.output.status.invalidCount = graph.createDataView(props.output.summary.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(() => new GPUSpatialScanStatistic(props)).toThrow(/overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('scan pipeline count stays constant as replicate batches grow', () => {
  const record = (permutations: number) => {
    const {device, graph, props} = createScene('poisson', permutations);
    try {
      return getGPUCommandNodePipelineKeys(
        new GPUSpatialScanStatistic(props).getCommandNodes(graph)
      ).size;
    } finally {
      device.destroy();
    }
  };
  expect(record(3)).toBe(record(9));
});

it('production scan intervals cover upper-bound totals exactly with no wrapped endpoints', () => {
  const totals = Uint32Array.of(0, 1, 1023, 1024, 1025, 0xffffff00, 0xfffffffd, 0xfffffffe);
  const output = new Uint32Array(totals.length * 1024 * 2);
  const source = `${SCAN_DRAW_INTERVAL_WGSL}
@group(0) @binding(0) var<storage, read> totals: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
 let index = invocation.x;
 let interval = getScanDrawInterval(totals[index / 1024u], 1024u, index % 1024u);
 output[index * 2u] = interval.first; output[index * 2u + 1u] = interval.stop;
}`;
  expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
  if (/workgroupBarrier|storageBarrier/.test(source))
    expect(findStatisticsBarrierRisks(source)).toEqual([]);
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [totals.length * 1024, 1, 1],
    {0: {0: totals, 1: output}}
  );
  for (const [domain, total] of totals.entries()) {
    let previous = 0;
    let covered = 0;
    for (let blockIndex = 0; blockIndex < 1024; blockIndex++) {
      const offset = (domain * 1024 + blockIndex) * 2;
      expect(output[offset]).toBe(Math.min(blockIndex * Math.ceil(total / 1024), total));
      expect(output[offset + 1]).toBe(Math.min((blockIndex + 1) * Math.ceil(total / 1024), total));
      expect(output[offset]).toBe(previous);
      expect(output[offset + 1]).toBeGreaterThanOrEqual(previous);
      expect(output[offset + 1]).toBeLessThanOrEqual(total);
      covered += output[offset + 1] - output[offset];
      previous = output[offset + 1];
    }
    expect(previous).toBe(total);
    expect(covered).toBe(total);
  }
});
